# SGLang DSV4.1 Engram All-to-All 解码、ROCm 独立 K/V 分配与 Sparse-MLA DCP、HiCache Page-Unified 端到端集成、vLLM MRV2 DBO+投机解码/DCP 与 TensorRT-LLM Anthropic 保活（Phase 130）

> **English Edition**: [sglang-engram-decode-dcp-mla-hicache-vllm-dbo-trtllm-phase130.en.md](sglang-engram-decode-dcp-mla-hicache-vllm-dbo-trtllm-phase130.en.md)  
> **Phase 130 (`priority: 139`) · 130 阶段 & 400 实验套件双里程碑 · Track 2（LLM 推理引擎与分离式运行时层）**  
> **证据等级**: `implementation` (一手上游提交源码) + `own-offline` (确定性 CPU 仿真，非 GPU 实测)

---

## 1. 背景与核心架构问题

在 DeepSeek-V4 / V4.1、GLM-5.3 与长上下文分离式推理引擎（SGLang、vLLM Model Runner V2、TensorRT-LLM）的生产演进中，2026-10-28 的 10 项上游提交解决了以下五个关键系统瓶颈：

1. **DeepSeek-V4.1 Engram 表在 Decode 与 CUDA Graph 下的集合通信放大**：当 Engram 表采用 `replication: none`（跨 TP 组分片）时，若在 Decode 阶段使用 `all_reduce` 或 `reduce_scatter`，每个 TP rank 的通信量随全局批大小与完整嵌入维度线性膨胀；切换为 `all_to_all` 并在模型构造期（已知并行拓扑与 GPU 型号后）解析 `lookup: auto`，可支持 CUDA Graph 捕获并显著降低跨卡流量。
2. **Rust 控制面与 API 反序列化开销**：`sglang-api-types` 此前在反序列化扁平 Token ID 列表（`TokenIds` / `StringListOrList`）时先将每个元素缓冲为中间 `Value` 枚举；同时在多 DP 副本部署下，Rust Server 需要通过集合通信收集所有 DP 监听端点（`rust_server_endpoints`），才能将控制请求扇出（fan-out）至全部 DP rank。
3. **ROCm 上融合 K/V 内存池导致的 Mooncake 同节点 PD 乱码与 AMD Sparse-MLA DCP 缺失**：CUDA 的 `trtllm_mha` `fmha_v2` prefill 依赖 V 位于 K 分配块内部的 int32 块偏移布局，但在 ROCm 上同节点 Mooncake PD 分离传输会因 V 嵌套在 K 分配内而解码出乱码；同时 AMD Triton sparse-MLA 后端此前缺乏 Decode Context Parallelism (DCP) 支持，且 AITER DSA decode 元数据未按实际活跃批大小 `bs` 切片 `kv_indptr`。
4. **HiCache `page_unified` 写回 JIT 模块探测错位与 Unified Memory + INT8 Mamba Checkpoint 冲突**：`page_unified` 与 `page_first` 虽共用头文件但属于不同模板实例化与 `cache_once` 键，用 `page_first` 探测会导致运行时崩溃；而 `--enable-unified-memory` 内存池不维护 INT8 Mamba 检查点池，必须显式拦截 `--enable-int8-mamba-checkpoint`。
5. **vLLM MRV2 双批重叠（DBO）与投机解码/DCP 的微批阈值下界，以及 TensorRT-LLM 长静默流 SSE 超时**：在 vLLM V2 Runner 中开启 DBO + 投机解码（如验证 3 个 draft token，即每请求 4 个 token）时，2 个微批至少需要 $2 \times (1 + 3) = 8$ 个 token，若 `dbo_decode_token_threshold = 7` 会导致微批无法覆盖完整请求；而在 TensorRT-LLM 服务 Anthropic Messages 流式接口时，超长 Prefill 或深度推理静默期若无 `ping` 事件保活，会被上游网关/LB 断开连接。

---

## 2. 一手源码证据与机制深挖（10 项上游提交）

| # | 仓库与提交 | 日期 | 核心机制与架构量化指标 |
|---|---|---|---|
| 1 | `sgl-project/sglang@1825a09` (`#46955`) | 2026-10-28 | DeepSeek-V4.1 (`dsv41`) Engram 层在 Decode 与 Piecewise CUDA Graph 下启用 `all_to_all` 查表，并将 `lookup: auto` 延迟到模型构造期 (`resolve_engram_config`) 根据并行布局与 GPU 架构解析；在 `TP=8, B_local=64, D=4096` 下单卡集合通信量比 `all_reduce` 降低 **16.0x**。 |
| 2 | `sgl-project/sglang@59af99a` (`#47008`) | 2026-10-28 | `sglang-api-codegen` / `sglang-api-types` 在反序列化 `OneOrMany` 扁平 ID 或字符串列表时，根据首元素类型直接流式解码为目标标量列表，避免缓冲为中间 `Value` 对象，每 Token 临时堆开销降低 **8.0x**。 |
| 3 | `sgl-project/sglang@69233a4` (`#46181`) | 2026-10-28 | `maybe_init_rust_server` 通过集合通信 `RustServer.gather_server_endpoints` 收集全部 DP 监听地址并注入 `rust_server_endpoints`，实现控制请求向所有 DP rank 的 **100%** 扇出覆盖。 |
| 4 | `sgl-project/sglang@8044ed4` (`#46972`) | 2026-10-28 | `handle_int8_mamba_checkpoint` 显式校验并在同时启用 `--enable-int8-mamba-checkpoint` 与 `--enable-unified-memory` 时抛出 `ValueError`（与 `--enable-lmcache` 拦截对齐）。 |
| 5 | `sgl-project/sglang@86f72e5` (`#46861`) | 2026-10-28 | `MHATokenToKVPool` 仅在 `k_shape == v_shape and not _is_hip` 时使用融合 K/V 分配；ROCm 上强制保持 K 与 V 独立分配，修复同节点 Mooncake PD 分离在 ROCm 上解码乱码的问题。 |
| 6 | `sgl-project/sglang@edd270f` (`#44322`) | 2026-10-28 | AMD Triton sparse-MLA 后端支持 DCP：新增 `@triton.jit def _dcp_pack_q_kernel`（打包 `[q_pe \| q_nope]` 并转换至输出 dtype），并将 `_sparse_mla_fwd_split_dim_kernel` 的 Q 步长从 `tl.constexpr` 改为运行时参数以适配 DCP 重排张量。 |
| 7 | `sgl-project/sglang@f65af1c` (`#39728`) | 2026-10-28 | 完成 HiCache `--hicache-mem-layout=page_unified` 端到端集成，新增 `can_use_page_unified_write_back_jit_kernel` 针对独立 `cache_once` 模板实例进行 JIT 可用性探测。 |
| 8 | `sgl-project/sglang@aa8e0a9` (`#41696`) | 2026-10-28 | `_prepare_aiter_dsa_decode_metadata` 中新增 `kv_indptr = kv_indptr[: bs + 1]`，将 AITER DSA decode 元数据限制在活跃批大小内（`bs=16, max_bs=256` 时长度从 `257 -> 17`，缩减 **15.12x**）。 |
| 9 | `vllm-project/vllm@cdd2750` (`#61464`) | 2026-10-28 | vLLM Model Runner V2 (`MRV2`) 支持 DBO（Dual-Batch Overlap）与投机解码及 DCP 联合运行：微批 token 阈值自动按 `max(threshold, 2 * (1 + num_spec_tokens))` 向上钳位（如 3 drafts 下 `7 -> 8`），且 target 模型按每微批槽位重建 prefill 元数据。 |
| 10 | `NVIDIA/TensorRT-LLM@461580b` (`#21030`) | 2026-10-28 | `tensorrt_llm/serve/anthropic_adapter.py` 与 `anthropic_protocol.py` 新增 `AnthropicPingEvent(type="ping")`，在流静默期间周期性发送 SSE keepalive ping，防止长 Prefill/思考流被网关超时切断。 |

---

## 3. 关键数学公式与确定性实验验证

### 3.1 vLLM MRV2 DBO + 投机解码微批 Token 阈值钳位公式

设投机解码每步验证 $K_{\text{spec}}$ 个草稿 token，则单个请求在一个验证步内携带 $1 + K_{\text{spec}}$ 个 token。在 $M_{\text{ub}} = 2$ 个微批的 DBO 流水线下，为保证每个微批至少容纳 1 个完整请求，有效阈值 $T_{\text{eff}}$ 必须满足：

$$T_{\text{eff}} = \max\!\bigl(T_{\text{cfg}},\; M_{\text{ub}} \cdot (1 + K_{\text{spec}})\bigr)$$

当 $K_{\text{spec}} = 3, M_{\text{ub}} = 2$ 时，$M_{\text{ub}}(1 + K_{\text{spec}}) = 8$，因此配置值 `(7, 8, 16)` 被确定性映射为 `(8, 8, 16)`。

`experiments/sglang_engram_decode_dcp_mla_hicache_vllm_dbo_trtllm_phase130/run.py --check` 全部 5 项不变量验证通过。

---

## 4. 解决方案架构师决策指南与可证伪问题

- **ROCm 上的 Mooncake PD 分离部署**：切勿在 ROCm 上复用针对 CUDA `trtllm_mha` `fmha_v2` 设计的单缓冲融合 K/V 内存池；K 与 V 独立分配是保证 RDMA/IPC 页指针正确寻址的前提。
- **可证伪后续问题**：在 DeepSeek-V4.1 `TP=8` 跨节点（4+4 GPU）部署下，Engram 表采用 `per_node` 复制 + 节点内 `peer_read` 与 `replication=none` + `all_to_all` 相比，在何种提示词命中率与批大小阈值下会出现 HBM 容量与跨节点 IB 延迟的盈亏平衡点？
