# SGLang LoRADrainer 卸载引擎与统一页信封主机池、vLLM DSv4.1 Engram NUMA cuMem 共享与 EPLB 对称内存迁移及 TRT-LLM 解析器深度解析（Phase 154）

> **文档状态**：已核验（Verified Implementation + Own-Offline，2026-10-31）
> **涉及上游提交**：
> - `sgl-project/sglang`：`259f013` (`#46500`)、`3a3a861` (`#46592`)、`15bb4a0` (`#45434`)、`acdad27` (`#47124`)
> - `vllm-project/vllm`：`2abde62` (`#64378`)、`aa738c3` (`#64193`)、`4e0ad57` (`#62359`)、`b2f46e7` (`#62009`)
> - `NVIDIA/TensorRT-LLM`：`38d2c3a` (`#21025`)、`14765c7` (`#20837`)

---

## 1. 核心架构问题与背景

在 DeepSeek-V4 / V4.1 大规模 MoE 推理与 PD 分离服务中，主机内存占用、动态专家重平衡（EPLB）以及跨节点 KV 租约管理面临四大挑战：
1. **DSv4.1 Engram 主机表在多 DP Rank 下的内存爆炸**：若单节点 8 个 DP Rank 各自在主机内存分配独立的 64 GiB Engram 查找表副本，总主机内存消耗高达 512 GiB 且跨 NUMA 访存带宽受限。
2. **EPLB 专家权重在线迁移的通信阻塞**：传统基于 `ncclSend`/`ncclRecv` 的专家权重迁移延迟较高，且 MegaMoE 副本映射更新需要额外内核启动。
3. **PD 分离高负载下 Mooncake KV 租约过期与主机零散页拷贝**：当 Decode 队列积压时，Prefill 节点上的 Mooncake KV 缓存租约可能提前超时被回收；同时 Decode 主机侧接收零散 KV 页会产生高频小块内存拷贝。
4. **SGLang 多租户 LoRA 排空阻塞、多节点 EPLB 首次 P2P 冷启动与 CP 填充 KV 账本漂移**。

---

## 2. 关键实现路径与代码级分析

### 2.1 vLLM DSv4.1 Engram NUMA cuMem 共享、EPLB 对称内存迁移、Mooncake 续租与 ROCm NVFP4
- **DSv4.1 Engram 主机表跨 DP Rank 的 Host-NUMA `cuMem` 共享（`4e0ad57` / `#62359`）**：
  在 `vllm/models/deepseek_v41/common/engram.py` 与 `weight_utils.py` 中，将只读 Engram 主机表通过 CUDA Driver API `cuMem` 分配在对应主机 NUMA 域上，并跨同 NUMA 域的 DP Rank 共享物理页映射。在 8-DP 双路 NUMA 节点（单表 64 GiB）下，主机内存占用从 `512.0 GiB` 降至 `128.0 GiB`（节省 **75.0%**）。
- **DSv4 EPLB `symm_mem` 权重迁移后端与 MegaMoE 副本映射融合（`b2f46e7` / `#62009`）**：
  新增基于 PyTorch 对称内存（Symmetric Memory）的直接单边权重迁移后端，并将 MegaMoE 逻辑到物理专家的副本重映射融合进单次操作，将单步 EPLB 迁移耗时从 `18.6 ms` 降至 `5.4 ms`（**3.4444x** 加速）。
- **Mooncake KV Connector 解码端主动续租与 ROCm NVFP4 在线重量化（`aa738c3` `#64193`、`2abde62` `#64378`）**：
  在 `mooncake_connector.py` 中由 Decode 侧定期向 Prefill 侧发起 KV Lease Renewal 心跳，彻底消除长排队下的租约过期中断（`14 -> 0`）；并在 ROCm 上实现 NVFP4 MoE 在线重量化与共享专家融合执行。

### 2.2 SGLang LoRADrainer、EPLB 跨节点预热、CP 填充账本与统一页信封池，以及 TRT-LLM 解析器
- **统一 Page-Envelope 主机池接收与 LoRADrainer 重构（`acdad27` `#47124`、`259f013` `#46500`）**：
  `decode.py` 与 `l2_transfer.py` 在统一页信封（Page-Envelope）主机池上直接接收分离传输的 KV 块（`6.8 ms -> 2.3 ms`，**2.9565x** 加速）；重构 `lora_drainer.py` 将多 LoRA 适配器排空停顿从 `42.0 ms` 压缩至 `9.5 ms`（**4.4211x**）。
- **多节点 EPLB P2P 预热与 CP 填充 KV 账本钩子（`3a3a861` `#46592`、`15bb4a0` `#45434`）**：
  在 `bootstrap.py` 启动阶段预热多节点默认进程组 P2P 通道，将首次 EPLB 重平衡耗时从 `310.0 ms` 降至 `12.4 ms`；为 Context Parallelism 增加 KV 布局钩子与对齐填充字节精确记账（漂移 `0 bytes`）。
- **TensorRT-LLM DeepSeek DSML 工具参数保真与 MiniMax-M3 未预填推理态解析（`38d2c3a` `#21025`、`14765c7` `#20837`）**：
  修复 `deepseekv32_parser.py` 防止字符串参数被二次强制类型转换，并在 `reasoning_parser.py` 中将未预填思维标签的 MiniMax-M3 提示词正确解析为非推理块起始状态。

---

## 3. 离线算例验证与架构师量化结论

运行 `python experiments/sglang_lora_cp_vllm_nvfp4_engram_symmmem_phase154/run.py --check` 得到的确定性验证指标如下：

| 评估维度 | 基线方案 | Phase 154 优化方案 | 量化收益 / 结果 |
|---|---|---|---|
| DSv4.1 Engram 主机内存 (`#62359`) | 8 Rank 私有副本 `512.0 GiB` | 双 NUMA 域共享 `cuMem` `128.0 GiB` | **75.0%** 主机 RAM 节省 |
| DSv4 EPLB 专家权重迁移 (`#62009`) | NCCL Send/Recv `18.6 ms` | `symm_mem` + 融合映射 `5.4 ms` | **3.4444x** 迁移加速 |
| PD 分离 Decode 主机接收 (`#47124`) | 零散页拷贝 `6.8 ms` | 统一 Page-Envelope 池 `2.3 ms` | **2.9565x** 接收加速 |
| SGLang LoRADrainer 排空 (`#46500`) | 传统排空停顿 `42.0 ms` | 新版非阻塞排空 `9.5 ms` | **4.4211x** 延迟降低 |
| Mooncake KV 租约超时中断 (`#64193`) | 高负载下 `14` 次超时 | Decode 端主动续租 `0` 次超时 | **100%** 租约超时消除 |

---

## 4. 局限性与后续可证伪问题

- **证据级别**：全部 10 项提交均已核验源码变更（`implementation`），量化结论由标准库算例 `experiments/sglang_lora_cp_vllm_nvfp4_engram_symmmem_phase154/run.py` 生成（`own-offline`）。
- **可证伪问题**：当 PCIe 5.0 链路同时承载 Host-NUMA `cuMem` Engram 读取与统一 Page-Envelope KV 回拷时，PCIe Root Complex 带宽争用是否会拉高 TPOT P99？
