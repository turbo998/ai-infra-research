# Megatron-LM CUDA-Graph 动态 Padding 归零、TE 编译态 FP8 权重缓存、LMCache SWA 前缀重放对齐与 Mooncake/NIXL 传输护栏

> **Phase 66 · Priority 75 · Track C（分布式训练、编译态 FP8、KV Cache 卸载与 RDMA/UCX 传输）**  
> **覆盖上游源码提交**：`NVIDIA/Megatron-LM@f9d9bdcc6e09`、`NVIDIA/Megatron-LM@92641a421f40`、`NVIDIA/TransformerEngine@9ea05cf2e8dc`、`LMCache/LMCache@a0ff4701284d`、`kvcache-ai/Mooncake@dae22d472e31`、`kvcache-ai/Mooncake@b66bb2b5f650`、`ai-dynamo/nixl@f3673f6be3f4`、`ai-dynamo/nixl@70badd6aad36`  
> **配套确定性实验**：`experiments/megatron-padding-zero-lmcache-swa-replay-mooncake-wrid/run.py` / `summary.json`

---

## 1. 架构背景与核心工程问题

在大规模混合专家（MoE）训练、CUDA Graph 量化动态推理以及跨节点 KV Cache RDMA/UCX 传输中，静态图捕获与动态运行态的边界不一致会导致严重的数值污染、越界写入或环形槽位覆盖：

1. **Megatron-LM CUDA Graph 重放时 Python 切片冻结导致 FP8/FP4 Padding 行污染全局 `amax`**：在动态推理（`DynamicInferenceContext`）中，CUDA Graph 在捕获阶段使用填满的 dummy tokens（此时 `padding_slice` 为空切片）。若在量化层前使用 Python 切片清零 padding 行，由于切片边界在捕获时被静态冻结为空，重放阶段当真实 token 数小于捕获上限（如 `real_token_count = 72 < 128`）时，行 `[72, 128)` 完全不会被清零，其中的残留/垃圾激活直接进入 FP8/FP4 GEMM 的 per-tensor `amax` 归约，严重压缩真实 token 的有效动态范围。
2. **TransformerEngine `torch.compile` 模式下 FP8 权重缓存（`is_first_microbatch`）无法跨函数式算子边界**：此前 `Linear` 在开启 FP8 权重缓存时直接回退到 eager 模式，因为首个 microbatch 原位更新缓存的工作区（workspace）违反了函数式自定义算子 `mutates_args=()` 的不可变契约。
3. **LMCache 在 vLLM 滑动窗口注意力（SWA）有界前缀重放（Prefix Replay，如 DeepSeek V4.1）下的全 Prompt 命中越界加载**：vLLM 开启 SWA 有界前缀重放时，调度器在分配 GPU KV 块之前会将外部连接器上报的命中 token 数向下取整到块边界（Block Boundary）。当 `12` token 的请求（`chunk_size = 4`）发生全命中时，LMCache 旧有的 `ret - 1 = 11` 规则会被 vLLM 向下取整并仅分配 `2` 个块（`8` 个槽位），而 LMCache 仍尝试异步取回 `[0, 12)` 共 `12` 个 token，直接越界溢出 `4` 个 token 并残留 `[8, 12)` 的查找锁！
4. **Mooncake TENT RDMA `sendNotification` 在 `ibv_post_send` 失败前递增 `wr_id` 导致在途槽位覆盖 & NVLink IPC 虚拟内存/容器 PID 冲突**：`RdmaEndPoint::sendNotification` 使用 `wr_id % 256` 索引发送缓冲区槽位；旧代码在调用 `ibv_post_send` 之前执行 `wr.wr_id = notify_send_wr_id_++`，若前 `255` 个通知（`wr_id 0..254`）仍在途未完成且第 `256` 次调用 `ibv_post_send` 失败返回 `EINVAL`，`notify_send_wr_id_` 已变为 `256` 而 `notify_pending_count_` 仍为 `255`，下一次重试将分配 `wr_id = 256`（即槽位 `256 % 256 = 0`），直接覆写仍在途的 `wr_id = 0` 缓冲区！此外，PyTorch `expandable_segments:True`（`cuMemCreate` VMM 分配）或跨容器独立 PID 命名空间会导致 `IntraNodeNvlinkTransport` 报出晦涩的 `cudaIpcGetMemHandle`/`cudaIpcOpenMemHandle` 错误。
5. **NIXL UCX 远端通知缺少请求执行标签（`treq`）与重投递（Repost）/释放时的遥测漏计**：UCX 接收端收到通知时仅有 26 字节 `TraceContext`（`tctx`），缺少发起端的 `(request_id, execution_number)` 关联；且当传输完成首次在 `postXferReq` 重投递或 `releaseXferReq` 中被观测到时，`nixl_telemetry` 漏计了完成统计，而因 `NIXL_IN_PROG` 被拒绝的重投递却提前重置了在途计时器。

---

## 2. 第一手源码实现路径与关键修复机制

### 2.1 Megatron-LM 设备指针 Padding 归零内核与全局 Batch K3 分位数均衡（`Megatron-LM@f9d9bdcc6e09`、`Megatron-LM@92641a421f40`）

- **CUDA Graph 安全的 `zero_padding_rows_kernel`**：在 `megatron/core/inference/contexts/zero_padding_rows_kernel.py` 与 `dynamic_context.py` 中，新增 `zero_padding_tokens(tensor, sequence_parallel_rank)`。Triton 内核 `_zero_padding_rows_kernel` 不接收 Python 标量边界，而是传入固定设备地址的 `int32[1]` 张量指针 `gpu_view.real_token_count`，在 GPU 端实时加载 `real_token_count = tl.load(real_token_count_ptr)`，并对满足 `rows + first_token_idx >= real_token_count` 的行执行只写归零（不读取原激活）。在 `captured_rows = 128`、`replay_real_tokens = 72`、真实 `amax = 2.0`、残留垃圾 `amax = 64.0` 的算例中，设备指针归零将 FP8 E4M3 缩放因子从 `7.0` 恢复至 `224.0`（**恢复 `32.0x` 有效量化精度**），且支持 Sequence Parallel 分片偏移 `first_token_idx = sequence_parallel_rank * tensor.shape[0]`。
- **全局 Batch K3 分位数路由均衡（Quantile Balancing）**：在 `megatron/core/transformer/moe/{router.py,moe_utils.py}` 与 `finalize_model_grads.py` 中，当 `qb_estimation_scope == "global_batch"` 时，每个专家路由器维护非持久化的 `int32` 直方图缓冲区 `qb_histogram [num_experts, num_bins]` 与 `qb_bin_bounds [-1.0, 1.0]`，在每个全局 batch 结束时通过 `_update_router_expert_bias_with_quantile` 将所有 MoE 层的直方图堆叠并在 `reduction_groups` 上执行单次批量 `all_reduce`，随后一次性更新 `expert_bias` 与 `qb_bin_bounds` 并调用 `mark_qb_bin_bounds_validated(bounds)`。

### 2.2 TransformerEngine `torch.compile` 下的 FP8 权重工作区缓存（`TransformerEngine@9ea05cf2e8dc`）

- 在 `transformer_engine/pytorch/module/linear.py` 中，移除 `if self.cache_weight and self.fp8: return "FP8 weight caching (is_first_microbatch)"` 的编译阻断（仅当使用 GPU 控制的 `skip_fp8_weight_update` 张量时回退 eager）。
- 在编译态（`use_compiled_op`）的首个 microbatch（`is_first_microbatch=True`）时，刻意将传入函数式算子的 `weight_workspace` 设为 `None`，使算子分配并返回全新的 FP8 权重工作区并存入 `self._fp8_workspaces["weight"]`；在随后的 microbatches（`1..M-1`）中传入已缓存的 `weight_workspace` 直接复用。在 `M = 8` 个 microbatch 的梯度累积步中，FP8 权重量化转换次数从 `8` 次降至 `1` 次（**消除 `87.50%` 重复权重量化**；4 层 4096 宽 Linear 峰值显存为 eager `452 MiB` / compiled `539 MiB`）。

### 2.3 LMCache SWA 有界前缀重放的块对齐回退与锁释放（`LMCache@a0ff4701284d`）

- 在 `lmcache/integration/vllm/lmcache_mp_connector.py` 中，新增 `get_prefix_replay_tokens(kv_cache_config)` 读取各 KV Cache 组的 `prefix_replay_tokens`。
- 当全 Prompt 命中（`ret == len(request.all_token_ids)`）且 `self._prefix_replay_tokens > 0` 时，不再执行 `need_to_load - 1`，而是按 `chunk = self.scheduler_adapter.lmcache_tokens_per_chunk` 向下对齐到整块边界：
  $$\text{aligned} = \max\left(0, \left\lfloor \frac{\text{ret} - 1}{\text{chunk}} \right\rfloor \times \text{chunk}\right)$$
  同时将未加载的尾部区间 `(aligned, ret)` 加入 `tracker.pending_free_ranges`（由 `free_lookup_locks(extra_ranges=...)` 释放锁），并将 `tracker.num_lmcache_hit_tokens` 更新为 `aligned`。在 `12` token 请求（`chunk = 4`）下，`aligned = 8` 与 vLLM 分配的 `2` 个块（`8` 槽位）严格吻合，**缓冲区溢出从 `4` tokens 降至 `0` tokens**。

### 2.4 Mooncake RDMA `notify_send_wr_id_` 提交后递增、NVLink 诊断与 NIXL `treq` 协议（`Mooncake@dae22d472e31`、`Mooncake@b66bb2b5f650`、`nixl@f3673f6be3f4`、`nixl@70badd6aad36`）

- **Mooncake `sendNotification` 槽位安全**：在 `mooncake-transfer-engine/tent/src/transport/rdma/endpoint.cpp` 中，将 `wr.wr_id = notify_send_wr_id_++;` 拆分为先赋值 `wr.wr_id = notify_send_wr_id_;`，仅当 `ibv_post_send()` 成功返回 `0` 后才执行 `++notify_send_wr_id_;`。当 `255` 个 WR 在途且第 `256` 次投递失败时，`notify_send_wr_id_` 保持为 `255`，重试仍使用空闲槽位 `255` 而不会回绕覆写槽位 `0`。
- **Mooncake NVLink VMM 与容器 PID 命名空间诊断**：在 `intranode_nvlink_transport.cpp` 中，当 `cudaIpcGetMemHandle` 失败时调用 `cuMemRetainAllocationHandle` 探测是否为 `cuMemCreate` 虚拟内存分配（直指 PyTorch `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`）；当 `cudaIpcOpenMemHandle` 返回 `cudaErrorDeviceUninitialized` 时，明确诊断跨容器独立 PID 命名空间导致的导出进程 PID 碰撞，指导配置 `--ipc=host --pid=host`（或 Kubernetes `hostPID` / `shareProcessNamespace`）。
- **NIXL UCX 16 字节 `treq` 通知标签与重投递遥测**：在 `src/plugins/ucx/ucx_backend.cpp` 中，当 `trace_context` 存在且 `request_id != 0` 时，在 26 字节 `tctx` 记录之后追加 16 字节小端序 `treq`（`struct requestTag { uint64_t requestId; uint64_t execution; }`），使接收端 `nixl::notif.received` 完整还原发起端请求与执行序号；在 `src/core/nixl_agent.cpp` 中，`postXferReq` 将新计时器暂存在局部变量 `start_time`/`timer` 中，仅在确认旧请求非 `NIXL_IN_PROG` 后才写入 `req_hndl->timer`，并在 `postXferReq` 重投递和 `releaseXferReq` 首次观测到 `NIXL_SUCCESS` 时补齐 `updateRequestStats(..., NIXL_TELEMETRY_FINISH)`。

---

## 3. 确定性实验结果与对比矩阵

运行 `python experiments/megatron-padding-zero-lmcache-swa-replay-mooncake-wrid/run.py --check` 得到以下验证指标：

| 评估维度 | 修复前基线 (Legacy) | 修复后方案 (Phase 66) | 关键量化收益 / 架构保证 |
| :--- | :--- | :--- | :--- |
| **Megatron CUDA Graph Padding 归零 (`72/128` 真实行)** | Python 切片冻结为空（`0` 行归零） | GPU 读取 `real_token_count[0]` 归零 `56` 行 | 恢复 **`32.0x`** FP8 有效量化缩放精度 |
| **TE `torch.compile` FP8 权重缓存 (`M=8` microbatches)** | 回退 eager 或每步重复量化 `8` 次 | 首 microbatch 新建工作区，后 `7` 步复用 | 权重量化开销降低 **`87.50%`** |
| **LMCache SWA 前缀重放全命中 (`12` tokens, `chunk=4`)** | 返回 `11` -> vLLM 分 `8` 槽位，取回 `12`（溢出 `4`） | 对齐至 `8` tokens 并释放 `[8, 12)` 锁 | 越界溢出 **`0` tokens**，零锁泄漏 |
| **Mooncake RDMA 通知槽位 (`255` 在途 + 投递失败)** | `wr_id` 增至 `256`，重试覆写槽位 `0` | 仅在 `ibv_post_send` 成功后递增 `wr_id` | 重试落入空闲槽位 `255`，**零在途覆写** |
| **NIXL UCX 通知追踪 & 遥测完成计数** | 仅 26B `tctx`；被拒重投递重置计时器 | 26B `tctx` + 16B `treq`；延迟提交计时器 | 跨节点端到端关联 + `100%` 遥测闭环 |

---

## 4. 面向解决方案架构师的设计练习与可证伪问题

1. **排障练习**：在 Kubernetes 集群的单机 8 卡节点上部署两个独立 Pod 运行 Mooncake `IntraNodeNvlinkTransport`，发现只有两个 Pod 中相同 `LOCAL_RANK`（例如同为 Worker 0，容器内 PID 均为 `128`）的 GPU 对在执行 `cudaIpcOpenMemHandle` 时报 `invalid device context`（`cudaErrorDeviceUninitialized`），而不同 Worker 编号之间通信正常。根本原因是什么？如何修复 Pod Spec？  
   - **解答**：CUDA IPC 句柄内嵌了导出进程的 PID。当两个容器拥有独立的 PID 命名空间时，双方 Worker 0 恰好在各自容器内获得相同的 PID，导入方 CUDA 驱动误认为该句柄是由本进程自身导出的，从而拒绝打开并返回 `cudaErrorDeviceUninitialized`。修复方法是在 Pod Spec 中同时开启 `hostIPC: true` 和 `hostPID: true`（或在同一 Pod 内启用 `shareProcessNamespace: true`）。
2. **可证伪假设**：在 LMCache 与 vLLM 集成时，若模型未启用滑动窗口前缀重放（`prefix_replay_tokens == 0`），对于 `12` token 的全命中请求，`LMCacheMPConnector` 是否也会回退整个末位块（仅加载 `8` 个 token）？  
   - **验证结论**：**不会**。当 `self._prefix_replay_tokens == 0` 时，代码仍走 `need_to_load = max(0, need_to_load - 1) = 11`，vLLM 分配 `3` 个块（`12` 槽位），LMCache 完整取回 `12` 个 token。
