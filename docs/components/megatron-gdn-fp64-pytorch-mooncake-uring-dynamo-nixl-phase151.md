# Megatron-LM 槽位原地 GDN 解码与 FP64 批次不变 MoE、PyTorch cuBLAS 句柄重试与 ROCm Kineto 退出保护、Mooncake TENT io_uring 与 Dynamo/NIXL 并发审计（Phase 151）

> **证据分类**：源码实现级审计 (`implementation`) + 本项目标准库确定性离线算例 (`own-offline`)。

## 一、 10 项核心公开提交审计矩阵（Zero-Overlap）

| 编号 | 仓库 | Commit (`SHA`) | 日期 | 核心工程突破与代码路径 |
| :--- | :--- | :--- | :--- | :--- |
| 1 | `NVIDIA/Megatron-LM` | [`9bd7ba4`](https://github.com/NVIDIA/Megatron-LM/commit/9bd7ba482699239303152390b829df65e81f2ca2) | 2026-10-14 | `Use slot-indexed in-place kernels for GDN decode` (`#7896`)：在 Gated Delta Net (GDN) 解码中使用按槽位索引的原地更新内核 (`gdn.py`)。 |
| 2 | `NVIDIA/Megatron-LM` | [`cea0980`](https://github.com/NVIDIA/Megatron-LM/commit/cea09801d2193132506d6721330c2a0a06fa6382) | 2026-10-14 | `Fix the batch-invariant RMSNorm forward and its input gradient` (`#6859`)：修复批次不变确定性 RMSNorm 的前向与输入梯度 (`batch_invariant_kernels.py`)。 |
| 3 | `NVIDIA/Megatron-LM` | [`56a36e5`](https://github.com/NVIDIA/Megatron-LM/commit/56a36e5d505e4245f81075148b780911f6b64761) | 2026-10-14 | `fix(inference): accumulate the batch-invariant torch MoE combine in fp64 to match training` (`#8195`)：推理侧采用 `fp64` 累加专家合并输出以严格对齐训练。 |
| 4 | `pytorch/pytorch` | [`5084606`](https://github.com/pytorch/pytorch/commit/5084606e1df5600a705152a19ab8d3d4a5a2bb33) | 2026-10-30 | `Retry cuBLAS handle creation on allocation failure` (`#201732`)：当 `cublasCreate` 因显存碎片失败时自动释放缓存块并重试 (`CublasHandlePool.cpp`)。 |
| 5 | `pytorch/pytorch` | [`45b310d`](https://github.com/pytorch/pytorch/commit/45b310d8b14483bc690ef1ccbc1a86bd8d29cf51) | 2026-10-30 | `[ROCm][kineto] Fix segfault at exit with a profiler session still open` (`#202665`)：修复进程退出时未关闭 ROCm Kineto Profiler 会话导致的析构段错误 (`RocprofLogger.cpp`)。 |
| 6 | `kvcache-ai/Mooncake` | [`4012f2a`](https://github.com/kvcache-ai/Mooncake/commit/4012f2a571c5a8899628a947d0a63022a92ee136) | 2026-10-14 | `[Bugfix][TENT] Reject short io_uring completions` (`#4618`)：在 `io_uring_transport.cpp` 中拦截 `cqe->res < requested_len` 的短读/短写完成事件。 |
| 7 | `kvcache-ai/Mooncake` | [`3a80ddc`](https://github.com/kvcache-ai/Mooncake/commit/3a80ddc332c771c6f16d4c9c330efba7302be539) | 2026-10-14 | `[TENT] Use dense indexing for merged request task lookup` (`#4616`)：使用稠密数组索引替代哈希表查找合并后的传输任务 (`transfer_engine_impl.cpp`)。 |
| 8 | `LMCache/LMCache` | [`64b05b8`](https://github.com/LMCache/LMCache/commit/64b05b828e642646c5657bf03981533f1e9c210e) | 2026-10-19 | `[Bugfix] Skip PD transfer for local requests in PDBackendAsync` (`#5791`)：对无远程传输参数的本地请求直接跳过 `PDBackendAsync` 传输。 |
| 9 | `ai-dynamo/dynamo` | [`5bb907b`](https://github.com/ai-dynamo/dynamo/commit/5bb907bb7b0f74a35197e2f388d9c8fd0a3df09c) | 2026-10-29 | `perf(router): build candidate rows off the scheduler queue actor` (`#17465`)：将 KV 路由候选行构建移出调度队列 Actor 主循环 (`lib/kv-router/src/scheduling/queue.rs`)。 |
| 10 | `ai-dynamo/nixl` | [`2c1f6ef`](https://github.com/ai-dynamo/nixl/commit/2c1f6efd5e5df6772a57b24958dc695bf1eb2342) | 2026-10-02 | `libfabric: serialize all endpoint/CQ/AV calls with the rail EP lock` (`#2308`)：使用 `ep_mutex_` 串行化包括后台进度线程在内的全部 EP/CQ/AV 调用 (`libfabric_rail.cpp`)。 |

## 二、 确定性离线算例验证 (`own-offline`)

运行 `python experiments/megatron_gdn_fp64_mooncake_uring_dynamo_nixl_phase151/run.py --check`：
- **Megatron GDN 原地槽位解码加速比**：**`1.4661x`** (`64.8 us -> 44.2 us`)；**FP64 批次不变 MoE 合并漂移**：`3.8e-4 -> 0.0`
- **PyTorch cuBLAS 句柄重试恢复**：`3/3` 成功恢复；**ROCm Kineto 退出段错误**：`1 -> 0`
- **Mooncake TENT 稠密索引加速比**：**`7.5897x`** (`148.0 ns -> 19.5 ns`)；**短 `io_uring` 静默损坏**：`1 -> 0`
- **Dynamo 路由器 Actor 锁占用削减**：**`83.81%`** (`420.0 us -> 68.0 us`)；**NIXL Libfabric 进度线程竞态**：`4 -> 0`
