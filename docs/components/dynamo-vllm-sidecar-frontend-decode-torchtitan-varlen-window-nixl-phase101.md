# NVIDIA Dynamo vLLM Sidecar 前端统一解码、TorchTitan `VarlenAttentionMetadata` 滑窗统一契约与 NIXL/LMCache 传输生命周期分析（Phase 101）

- **日期 / 阶段**：2026-10-27 · Phase 101 (`priority: 110`)
- **组件归属**：`interconnect-networking`
- **关联问题 ID**：`dynamo-vllm-sidecar-frontend-decode-torchtitan-varlen-window-nixl-phase101`
- **关联证据 ID**：`claim-dynamo-sidecar-decode-torchtitan-varlen-nixl-phase101`
- **可复现实验套件**：[`experiments/dynamo_sidecar_torchtitan_varlen_nixl_phase101/`](../../experiments/dynamo_sidecar_torchtitan_varlen_nixl_phase101/run.py)

---

## 1. 架构背景与核心问题

在分离式推理网关（NVIDIA Dynamo）、分布式长序列训练（TorchTitan / Megatron-LM / TransformerEngine）以及 RDMA/存储传输层（NIXL / LMCache）中，存在四类关键的跨组件协议与生命周期问题：

1. **分离式推理网关中后端 Sidecar 与前端的重复解码分叉**：在 NVIDIA Dynamo 的 vLLM sidecar（`lib/sidecar/vllm/src/convert.rs`）中，过去通过 `requires_frontend_decode` 判断请求是否包含可见停止词 ID（`stop_token_ids_visible`）来决定由 vLLM sidecar 还是 Rust 前端执行文本解码。双路径分叉不仅在每个生成步上冗余序列化 UTF-8 文本块，还导致停止词截断与首包 `prompt_logprobs` 语义不一致。
2. **变长打包序列（Varlen Attention）与 FlexAttention 的滑窗元数据分裂**：在 TorchTitan 中，全注意力（Full Attention）与滑动窗口注意力（Sliding-Window Attention）过去通过不同配置传递窗口参数，未能与 `FlexAttention` 的元数据驱动掩码契约对齐，增加了 Ulysses Context Parallelism（CP）下混合注意力层的校验复杂度。
3. **共享节点上 NIXL Initiator 固定默认端口冲突与未锚定 `ss` 端口探测误判**：`nixl_test initiator` 过去未显式设置 `cfg.listenPort`，回退到固定默认端口 `8888`，在多任务共享节点上频繁触发 `Address already in use`（退出码 `134` SIGABRT）；同时 `.ci/scripts/common.sh` 中 `ss -tuln | grep -q :$next_port` 会将对端端口或前缀端口（如 `:88880`、`:88881`）误判为占用。
4. **LMCache HF3FS 原生 L2 适配器后台 Demux 线程退出崩溃与 XLA/NCCL 通信器拆分死锁**：在 LMCache 的 HF3FS L2 适配器测试夹具中，仅关闭 `client` 而不调用 `adapter.close()` 会遗留后台 demux 线程并在 Python 解释器退出时触发 `abort`；在 TransformerEngine JAX 分布式融合注意力中，XLA/NCCL 缓存通信器替换（communicator splitting）会引发进程挂起。

---

## 2. 一手源码实证分析（8 个零重叠上游提交）

### 2.1 NVIDIA Dynamo `#17324` (`98f74e1dc8`)：vLLM Sidecar 无条件前端解码收敛

在 `lib/sidecar/vllm/src/convert.rs` 与 `lib/sidecar/testkit/tests/support/vllm.rs` 中：
- **移除 `requires_frontend_decode` 分支**：删除基于 `stop_conditions.stop_token_ids_visible` 的条件判断，在 `build_generate_request` 中统一将原生 protobuf 响应选项设为 **`output_text = Some(false)`** 且 **`output_token_ids = true`**。
- **兼容性信封（Compatibility Envelope）清理**：`skip_special_tokens` 在兼容信封中被接受但不再下发给 sidecar（由前端统一处理解码与特殊 token 过滤）。
- **协议收益与首包语义保证**：
  - 每步生成仅在 sidecar 与网关前端之间传输 4 字节 `token_id`（省去平均 ~6 字节 UTF-8 文本片段 + ~4 字节 protobuf 字符串封装开销），单步流式有效载荷减少 **`71.43%`**（从 `14 B` 降至 `4 B`）。
  - 首个输出包（first output）在前端执行任何停止字符串（stop string）截断之前完整携带 `prompt_logprobs` 与 `prompt_candidates`。

### 2.2 TorchTitan `#5607` (`b036f21aac`)、Megatron-LM `#8404` (`55561e5ef2`) 与 TransformerEngine `#3681` (`a7b6c0cbd2`)

- **TorchTitan `VarlenAttentionMetadata` 滑窗统一契约 (`#5607`)**：
  - 将 `window_size = (left, right)` 移入 `VarlenAttentionMetadata`，使全注意力（`window_size=(-1, 0)`）与滑动窗口注意力（`SlidingWindowVarlenInnerAttention` / `UlyssesSlidingWindowVarlenInnerAttention`）共享同一前向执行路径，并与 `FlexAttention` 元数据契约完全对齐。
  - 在 `test_context_parallel_validation.py` 与 `test_muse_spark_gb300_recipe` 中验证了纯滑窗模型（`first_full_attention is None`）在忽略负载均衡器时仍严格校验注意力头整除性。
- **TransformerEngine JAX NCCL Communicator Splitting 死锁规避 (`#3681`)**：针对 XLA/NCCL 缓存通信器替换在 L1 分布式融合注意力（`test_distributed_fused_attn.py`）中触发的死锁，在注意力进程级显式禁用 NCCL communicator splitting。

### 2.3 NIXL `#2456` (`744fc8aca1`) / `#2416` (`14145430cf`) 与 LMCache `#6052` (`d0a8680792`) / `#6037` (`21fecfd8a0`)

- **NIXL Initiator OS 动态端口绑定与锚定 `ss` 探测 (`#2456`)**：
  - 在 `test/nixl/nixl_test.cpp` 的 `runInitiator()` 中显式设置 **`cfg.listenPort = 0`**，由操作系统内核分配空闲临时端口，消除固定 `8888` 端口引发的并发 `bind` 失败与进程 `SIGABRT`（exit 134）。
  - 将 `.ci/scripts/common.sh` 中的端口探测升级为 `ss -Htuln | awk '{print $5}' | grep -qE "[:.]${next_port}$"`，仅匹配第 5 列（本地地址:端口）且严格锚定行尾，消除对端端口与前缀端口误报。
  - `#2416` 将 `cu13` wheel 构建升级至 CUDA `13.4.2` 并独立锁定 PyTorch 索引。
- **LMCache HF3FS L2 适配器 Demux 线程安全关闭 (`#6052`) 与 Rust `raw-block` 扩展 (`#6037`)**：
  - `#6052` 在 `hf3fs_l2_adapter` fixture 的 `finally` 块中显式调用 `adapter.close()`，确保 `NativeConnectorL2Adapter` 的后台 demux 线程在解释器退出前干净终止。
  - `#6037` 在 `setup.py` 中新增 `lmcache[raw-block]`（安装 `lmcache-rust-raw-block-io`），为 L2 裸块 I/O 提供零拷贝 Rust 扩展入口。

---

## 3. 确定性实验结果总结

运行 `python experiments/dynamo_sidecar_torchtitan_varlen_nixl_phase101/run.py --check` 得到以下密封实验指标：

| 评估维度 | 核心指标 | 实测 / 仿真结果 |
|---|---|---:|
| Dynamo vLLM Sidecar 前端解码 | 单步流式载荷缩减率（128 并发 $\times$ 512 tokens 总流量） | **`71.43%` (`14 B -> 4 B`, `896 KiB -> 256 KiB`)** |
| TorchTitan `VarlenAttentionMetadata` | 打包序列 (`[4K, 8K, 16K, 32K]`) 在 `window_size=(1024, 0)` 下计算分块缩减率 | **`91.48%` (`43,776 -> 3,728` 活跃分块)** |
| NIXL Initiator 端口与 `ss` 探测 | 未锚定 `grep :8888` 误报数 vs `awk '{print $5}'` 行尾锚定误报数 | **`3` 次误报 $\to$ `0` 次误报（冲突率 `0.0%`）** |
| LMCache HF3FS Native L2 Adapter | `try / finally: adapter.close()` 后台 Demux 线程残留率 | **`0.0%`（干净回收）** |

---

## 4. 解决方案架构师设计练习与可证伪问题

1. **打包变长序列滑窗分块复杂度核算**：给定打包文档长度序列 $[L_1, L_2, \dots, L_k]$ 与分块大小 $B = 128$，写出 `VarlenAttentionMetadata(window_size=(W, 0))` 下总活跃块数 $\sum_i T_{\text{active}}(L_i, W)$ 的闭式表达式，并解释为何跨文档边界不会因为共享同一个前向内核而发生键值泄漏。
2. **RDMA/TCP 控制面临时端口池容量评估**：在单台 8-GPU 节点并发启动 64 个 NIXL Initiator 实例时，对比固定默认端口（`listenPort = 8888`）与内核临时端口池（`listenPort = 0`, `net.ipv4.ip_local_port_range = 32768 60999`）的绑定成功概率。
