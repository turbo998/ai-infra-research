# Megatron-LM MFSDPv2 分阶段 MXFP8 平面聚合、TorchTitan GB300 128 卡 RL 拓扑与 Flux 潜变量调度、LMCache 回调式 Lookup、Mooncake 单次时钟读取与 NIXL/Dynamo 原子隔离深度解析

> **语言 / Language**: [中文](megatron-mfsdp-mxfp8-planes-lmcache-lookup-nixl-ring.md) | [English](megatron-mfsdp-mxfp8-planes-lmcache-lookup-nixl-ring.en.md)
> **对应问题**: `megatron-mfsdp-mxfp8-planes-lmcache-lookup-nixl-ring` (`priority: 76`, Phase 67 Track C)
> **证据等级**: `implementation` + `author-benchmark` + `own-offline`

---

## 1. 核心架构问题与背景

在超大规模分布式训练（MXFP8 FSDP、816B RL 训练）与分离式 KV Cache 传输/遥测控制面中，以下五项机制直接决定了通信效率、显存安全与尾延迟：

1. **Megatron-LM MFSDPv2 在 MXFP8 下的冗余双平面 All-Gather 与 TE 容量越界**：
   - MXFP8 权重同时包含行向平面 (`_rowwise_data` + `_rowwise_scale_inv`) 与列向平面 (`_columnwise_data` + `_columnwise_scale_inv`)。旧版 MFSDPv2 在每次 Unshard 时无条件同时 All-Gather 两个平面，即便前向传播 (`Y = X * W^T`) 只需行向平面、无激活重计算的反向传播 (`dX = dY * W`) 只需列向平面。
   - 同时，TransformerEngine 的 Triton `_row_id_map_pass_2_kernel` 在开启 `drop_and_pad` 且路由 Token 数超出 `num_out_tokens` 容量时未截断 `row_id >= num_out_tokens`，导致越界写坏后续内存。
2. **TorchTitan Muse Spark 816B-A42B GB300 RL 训练的跨机架专家 All-Gather 与 Flux 步长移位偏差**：
   - 在 GB300 上使用 `256` 卡 (`FSDP=64 x TP=4 x EP=4`) 运行 816B-A42B RL Trainer 时，专家权重的 FSDP 分片组跨度达 `64` 个 Rank（跨越多台主机机架），限制了 `131,072` 长上下文 MFU；而在 Flux 扩散模型中，`get_schedule` 在 `pack_latents` (2x2 Patch 打包) 之前被调用，传入了比实际 Transformer 序列长度大 `4x` 的未打包 `latent_height * latent_width`。
3. **LMCache 多存储节点异步 Lookup 的乱序竞态与尾锁泄漏**：
   - 旧版 `LMCacheAsyncLookupClient` 靠调度器轮询触发 `QUERY_PREFETCH_STATUS`，不仅引入 1 个调度步延迟，还可能使 `QUERY_PREFETCH_STATUS` 先于尚未完成的 `LOOKUP` 到达远端，引发假阴性 (`0` 命中) 与预取读锁泄漏；且当多台存储服务器返回不同命中块数时，超出 `min(hits)` 的尾部块读锁未被释放。
4. **Mooncake RDMA 链式发送的逐 WR 时钟系统调用开销**：
   - `RdmaEndPoint::submitPostSend` 在持有端点写锁 `lock_` 期间构造 WR 链并调用 `ibv_post_send`，但旧代码对链上每个 Slice 单独调用 `getCurrentTimeInNano()` (`clock_gettime(CLOCK_REALTIME)`，约 `19 ns/WR`)，占据了建链与持锁时间的大头并推迟了网卡 Doorbell。
5. **NIXL 共享内存环形缓冲的伪共享 (False Sharing) 与 Dynamo 逐 Token 全局原子计数竞争**：
   - NIXL `sharedRingBuffer` 的 `write_pos` 与 `read_pos` 紧邻在同一缓存行且每次 `push/pop` 都执行跨核原子读取；Dynamo `RequestMetrics` 在每个生成 Token 上对共享模型计数器执行 `fetch_add(1)`，在高并发流式输出下造成严重的跨核缓存行抖动。

---

## 2. 上游源码实现与关键机制分析

### 2.1 Megatron-LM MFSDPv2 分阶段 MXFP8 平面聚合 (`#7907`) 与 TE 修复 (`#3347`, `#3701`)

- **MFSDPv2 按计算阶段选择 MXFP8 平面 (`18bc49d2f6bc`)**：
  - `FsdpModule._unshard_events` 升级为 `dict[Phase, torch.cuda.Event]`，`QuantizedDBuffer.unshard(phase=...)` 在 `Phase.FORWARD` 仅 All-Gather 行向平面 (`rowwise_data` + `rowwise_scale_inv`)，在无重计算的 `Phase.BACKWARD` 仅 All-Gather 列向平面 (`columnwise_data` + `columnwise_scale_inv`)，将单阶段 MXFP8 All-Gather 通信量与非分片平面缓冲缩减 **50.0%**。
  - `update_mxfp8_tensor` 仅对本次新聚合的非空平面执行 Scale Padding，避免用未聚合的空平面覆盖另一阶段已填充好的 Scale。
- **TE Triton 容量丢弃与 GroupedTensor 视图/流追踪 (`c9c92255465b`, `2199b7166417`)**：
  - `_row_id_map_pass_2_kernel` 新增 `row_id = tl.where(row_id >= num_out_tokens, -1, row_id)`，将超出容量的 Token 标记为 `-1` 安全丢弃。
  - `GroupedTensor` 在构造时通过 `_make_Wrapper_subclass` + `view_as` 保留 Autograd `_base` 链，并实现 `record_stream` 透传至全部底层数据/Scale/Amax 张量。

### 2.2 TorchTitan：GB300 128 卡 RL 拓扑 (`#5581`) 与 Flux 打包潜变量调度 (`#5580`)

- **Muse Spark 816B-A42B GB300 RL Trainer 128 卡拓扑 (`43d1bb98014e`)**：
  - 从 `256` 卡 (`FSDP=64 x TP=4 x EP=4`) 重构为 **`128` 卡 (`FSDP=8 x TP=4 x CP=4 x EP=16`)**，将 `EP=16` 完整收敛在单个 4 主机 GB300 NVLink 域内，专家 FSDP 聚合组缩小 **`8.0x`** (`64 -> 8`)，并在全局与滑动窗口层全面启用 `UlyssesVarlenInnerAttention`（配合 `scratch_capacity_factor=16.0`），在 `131,072` 上下文下将 MFU 从 `15%` 提升至 **`28%`**（峰值显存 `186 GiB`）。
- **Flux `denoise` 打包序列长度修正 (`d681f48b43ad`)**：
  - 将 `get_schedule` 移至 `pack_latents` 之后并传入 `latents.shape[1] = (H/16)*(W/16)`（例如 `256px` 图像为 `256` 而非未打包的 `1024`），恢复了正确的扩散时间步偏移曲线。

### 2.3 LMCache (`#5960`, `#5922`)、Mooncake (`#4793`)、NIXL (`#2018`) 与 Dynamo (`#17067`, `#16106`)

1. **LMCache 回调驱动 `LookupState` 与多节点尾锁释放 (`ff9185d419e1`, `a81b4e5aea0c`)**：
   - 每个存储节点的 `LOOKUP` Future 完成时立即在回调中向该节点发送 `QUERY_PREFETCH_STATUS`（消除 1 步延迟与乱序竞态），通过 `scheduler_checked` 门控限制每个调度步最多重查 1 次，并对命中数高于 `min_hits` 的节点自动调用 `free_lookup_locks(hashes[min_hits:server_hits])` 释放多余尾块读锁；`PrefetchTaskSpec` 新增独立的 `allowed_l1_memory_tags` 与 `allowed_l2_memory_tags` 过滤。
2. **Mooncake RDMA 每条 WR 链单次读取时钟 (`dd1217b6c461`)**：
   - `RdmaEndPoint::submitPostSend` 在每条 QP 发送链循环外调用 1 次 `post_ts = getCurrentTimeInNano()` 并复用于链内所有 `slice->ts`。在 256-WR（2 条 128-WR 链）基准下，持锁建链耗时从 `6.3 us` 降至 **`1.9 us`**（每 WR 从 `25 ns` 降至 `7.5 ns`，**减少 69.84%**）。
3. **NIXL 256B 对齐环形缓冲 (`a9f1f7a76f46`) 与 Dynamo 64-Token 批量计数 (`906e89d3a304`)**：
   - NIXL `sharedRingBuffer` (`TELEMETRY_VERSION = 6`) 将 `write_pos` (偏移 `0`) 与 `read_pos` (偏移 `256`) 对齐至 `DESTRUCTIVE_INTERFERENCE_SIZE = 256` 字节（`sizeof(bufferHeader) == 512`），并在生产者/消费者本地缓存 `cachedReadPos_` / `cachedWritePos_`，仅在本地视图显示环满/环空时才跨核读取对端原子变量；同时用 `flock(LOCK_SH)` 检测存活消费者。
   - Dynamo `RequestMetrics` 将逐 Token 的 `output_tokens` 累加缓冲在请求本地 `pending_output_tokens` 中，每满 `LOCAL_FLUSH_TOKENS = 64` 个 Token（以及请求结束/Drop 时）才向全局模型计数器刷写一次，跨核原子争用降低 **`64.0x`**；并新增 `--freeze-gc-after-init` 在模型初始化后冻结 Python GC 堆。

---

## 3. 确定性实验验证 (`own-offline`)

运行 `python experiments/megatron-mfsdp-mxfp8-planes-lmcache-lookup-nixl-ring/run.py --check`：

| 验证维度 | 优化/修复前 | 优化/修复后 | 确定性结论 |
|---|---|---|---|
| MFSDPv2 MXFP8 单阶段 All-Gather (`4096x4096`) | `34,603,008 B` (双平面) | `17,301,504 B` (单平面) | 单阶段通信与平面缓冲 **减少 50.0%** |
| TorchTitan 816B GB300 RL Trainer | `256` GPUs (`FSDP=64, EP=4`), `15%` MFU | `128` GPUs (`FSDP=8, CP=4, EP=16`), `28%` MFU | Trainer GPU **减半**，专家 FSDP 组缩小 **`8.0x`** |
| LMCache 多节点 Lookup (`hits=[6, 4, 7]`, chunk=16) | 轮询延迟 + 超出块 `[64..112)` 锁泄漏 | 回调即发 + 释放 `srv0[64:96)`, `srv2[64:112)` | 匹配 `64` tokens，零尾块锁泄漏 |
| Mooncake 256-WR (`2x128`) 持锁建链耗时 | `256` 次时钟调用 / `6.3 us` | `2` 次时钟调用 / `1.9 us` | 建链持锁与 Doorbell 延迟 **降低 69.84%** |
| NIXL RingHeader & Dynamo 500-Token 全局原子写 | 同缓存行伪共享 / `500` 次全局原子加 | `512 B` 头 (`256 B` 对齐) / `8` 次刷写 | 7 次非满 Push 远端原子读降为 `0`，Dynamo 原子争用 **`64.0x` 缩减** |
