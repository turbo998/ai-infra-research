# Mooncake RDMA Auto-GID 7 级候选探测 (`PeerAutoGidStateTable`)、双端点故障确认状态机与有界在途 Scatter 重分片 (`MooncakeTransferEngineExecutor`) 机制

> **English Version**: [mooncake-rdma-auto-gid-reprobe-bounded-reshard.en.md](mooncake-rdma-auto-gid-reprobe-bounded-reshard.en.md) | **关联实验**: [`experiments/mooncake_gid_reprobe_reshard_sim/README.md`](../../experiments/mooncake_gid_reprobe_reshard_sim/README.md)

### 一句话结论

Mooncake 在 RDMA 传输引擎（commit `5ea6cf0d95`）与 Reshard 执行器（commit `0977fc7066`、Store commit `67df8c414a`）中引入了 **7 级 GID 候选分类器（`AutoGidCandidateClass`，将 RFC1918 `10/8` 私有 IPv4-mapped GID 独立提级至 `fe80::` Link-Local 之上）**、**按 `(peer, selection_epoch)` 隔离的双端点故障确认状态机（`PeerAutoGidStateTable::recordCompletion`：首端点故障返回 `kRetrySameGid`，仅当严格更新的 `endpoint_id > first_failed_endpoint_id` 再次故障时才触发 `kAdvanceGid` 并附带 `auto_gid_reprobe_from` 协商通告）**，以及 **`MooncakeTransferEngineExecutor` 基于 `max_inflight_batches` 的有界滑动窗口异步 Scatter 传输与 `_TransferSubmissionState.POISONED` 毒化隔离机制**，在 400GbE RoCEv2 多子网/容器 Overlay 混布环境下消除 GID 误选死锁，并将 64 GiB 跨节点模型重分片（Reshard）有效吞吐从单批同步的 `313.03 Gbps`（`1.7562 s`）提升至 4 批滑动窗口的 **`357.85 Gbps`（`1.5363 s`，提升 `+14.3%`，Pinned 窗口缓冲受控在 `1,024 MiB`）**。

### 组件责任与上游主线位置

| 模块与文件 | 上游仓库与 Commit / Blob SHA | 核心工程职责 |
| :--- | :--- | :--- |
| `mooncake-transfer-engine/include/transport/rdma_transport/rdma_gid_probe.h` | `kvcache-ai/Mooncake` commit `5ea6cf0d95d3a69b4ba196d28bf1f96e28ec5e10` / blob `eb0c5fc7710d425a510ece1fc651c452945c5176` | 定义 7 级 `AutoGidCandidateClass`、5 态 `AutoGidDataPathAction`、环形候选游标 `reselectAutoGidCandidate` 以及 `PeerAutoGidStateTable` 状态机。 |
| `mooncake-transfer-engine/src/transport/rdma_transport/rdma_endpoint.cpp` | `kvcache-ai/Mooncake` commit `5ea6cf0d95d3a69b4ba196d28bf1f96e28ec5e10` / blob `78916020bbd0284ec45f7546409677db2f8b41bf` | 在 `fillLocalHandshakeDesc` 中注入 `auto_gid_reprobe_from` 与 `ready_ack_supported` 握手字段，并在析构失败时通过 `.release()` 保留 DMA 通知缓冲区防 Use-After-Free。 |
| `mooncake-reshard/python/mooncake/reshard/transfer_engine/executor.py` | `kvcache-ai/Mooncake` commit `0977fc7066ece59f93440a1db59cb805906fe182` / blob `ec05a8ba8f19ab07c149ab92d3511432a831fc71` | 实现 `MooncakeTransferEngineExecutor._execute_batches` 有界 `max_inflight_batches` 队列、`submit_scatter_read/write` 票据校验与 `POISONED` 异常生命周期保护。 |
| `mooncake-store/src/client_service.cpp` | `kvcache-ai/Mooncake` commit `67df8c414ad04ec415f69da351e6fc92774e8a6f` / blob `fab103e035ed84ac0079a2bf343bb9db7898cd68` | 实现 `GetContiguousSliceRange` 连续 Slice 地址合并验证与 `GetTransportRegistrationLimit`（针对 `efa`/`cxi`/`ub` 协议的分段 MR 注册上限）。 |

### 核心实现路径、数据布局与同步契约

1. **7 级 RoCEv2 GID 候选分级：为什么必须将 RFC1918 私有 IPv4 从 Degraded 层拆分出来（`rdma_gid_probe.h:32-47`）**：
   - 在数据中心 RoCEv2 宿主机上，单张 `mlx5` 网卡的 GID 表同时包含公网/可路由 IPv6、RFC1918（`10.0.0.0/8`）IPv4-mapped GID、Docker/Calico 容器 Overlay 网段（如 `172.17.0.0/16`）以及 IPv6 Link-Local（`fe80::`）条目。
   - 早期把 RFC1918 私有 IPv4 与 `fe80::` 混在同一个 Degraded 等级，导致探测时可能优先选中排在前面的 `fe80::` GID（`fe80::` 无法跨越叶脊 ToR 三层路由，跨机架建连必挂）。
   - Commit `5ea6cf0d95` 将候选集精细划分为 7 个严格全序等级：
     1. `kNetworkRoutable = 0`（绑定 netdev 的全局可路由地址）
     2. `kNoNetworkRoutable = 1`（无 netdev 绑定的全局可路由地址）
     3. **`kNetworkPrivateV4 = 2`（绑定 netdev 的 RFC1918 / CGNAT IPv4-mapped GID，数据中心 `10/8` RoCE 主力网段）**
     4. `kNetworkDegraded = 3`（绑定 netdev 的 `fe80::` Link-Local 或容器 Overlay 网段）
     5. `kNoNetworkPrivateV4 = 4`
     6. `kNoNetworkDegraded = 5`
     7. `kFallbackNonzero = 6`
2. **双端点故障确认（Two-Endpoint Confirmation）与跨节点 `auto_gid_reprobe_from` 协同进位（`rdma_gid_probe.h:145-183, 299-350`）**：
   - 当某个 GID 在建连握手时看似通过，但在首个 RDMA 数据面 Write/Read WQE 时出现 CQE 超时（`IBV_WC_RETRY_EXC_ERR`），如何区分“瞬态拥塞丢包”与“GID 选错（如误选不通的 Overlay 网段）”？
   - `PeerAutoGidStateTable::recordCompletion(peer, attempt, success)` 采用严密的状态机：
     - **Epoch 过滤**：若 `attempt.selection_epoch != state.selection_epoch`，说明是旧连接残留的迟到 CQE，立即返回 `kIgnoreDuplicate`；
     - **首次端点故障（`state.first_failed_endpoint_id == 0`）**：记录 `first_failed_endpoint_id = attempt.endpoint_id`，置 `recovery_active = true`，但返回 **`kRetrySameGid`**（在同一 GID 上重建一次 QP 端点重试，过滤瞬态抖动）；
     - **同一或更早端点的重复报错（`attempt.endpoint_id <= state.first_failed_endpoint_id`）**：一个 QP 销毁前可能涌出多个失败 CQE，直接返回 `kIgnoreDuplicate` 防止把一次故障误计为多次；
     - **第二个新端点再次故障（`attempt.endpoint_id > state.first_failed_endpoint_id`）**：确认当前 GID 数据面不可达！置 `advance_pending = true, advance_requested_locally = true`，返回 **`kAdvanceGid`**。
     - **双向协同进位（Advisory Handshake）**：下一次 `prepare()` 重连时，本地通过 `reselectAutoGidCandidate` 推进到下一个候选 GID，分配新 `selection_epoch`，并在握手包 `HandShakeDesc.auto_gid_reprobe_from` 中填入刚才失败的 GID；对端收到后触发 `matching_advisory = (advisory_from == state.current.gid)`，即使对端本地预算 `remaining_advances == 0` 也会被临时授予 `state.remaining_advances = 1` 同步切换 GID！
3. **有界在途 Scatter 重分片与未决 DMA 毒化防护（`executor.py:355-515`）**：
   - 在 RL 训练向推理引擎同步权重（Mooncake Reshard）时，若一次性提交全部 Scatter 批次会耗尽网卡 WQE 队列与 Pinned 内存；若逐批同步等待（`max_inflight_batches = 1`）又会在每个批次尾部付出完整的 CQE 排空延迟。
   - `_execute_batches` 维护容量为 `max_inflight_batches` 的双端队列 `published: deque[_PublishedBatch]`：每当队列满（`len(published) >= max_inflight_batches`），先调用 `drain_published(published.popleft())` 排空最早提交的批次，再调用 `submit_scatter_read` / `submit_scatter_write` 提交新批次。
   - **防 DMA Use-After-Free 铁律**：一旦任何批次在提交或 `drain()` 时抛出 `TransferCompletionUnknownError` 或 `TransferCompletionInterrupted`，意味网卡硬件可能仍在后台异步读写该内存页，此时 `TransferSubmission` 立即转入 `_TransferSubmissionState.POISONED`，拒绝后续任何批次并阻止上层提前释放 `AllocationLifetimeToken`。

### 关键公式、单位与工程假设

1. **有界滑动窗口下的重分片有效吞吐与 Pinned 窗口内存上界**：
   $$T_{\text{reshard}}(W) = N_{\text{batches}} \cdot \left( \frac{B_{\text{batch}}}{\text{BW}_{\text{wire}}} + (t_{\text{fence}} + t_{\text{submit}}) \cdot \max\!\left(0.04,\, 1 - \min(1.0, 0.42(W - 1))\right) \right)$$
   $$M_{\text{window}}(W) = W \times B_{\text{batch}}, \qquad \text{Goodput}(W) = \frac{N_{\text{batches}} \cdot B_{\text{batch}} \cdot 8}{T_{\text{reshard}}(W)}$$
2. **双端点确认条件**：
   $$\text{Action}(e, \tau) = \begin{cases} \text{kIgnoreDuplicate}, & \tau \neq \tau_{\text{curr}} \lor (e \le e_{\text{first\_fail}} \land e_{\text{first\_fail}} > 0) \\ \text{kRetrySameGid}, & \tau = \tau_{\text{curr}} \land e_{\text{first\_fail}} = 0 \\ \text{kAdvanceGid}, & \tau = \tau_{\text{curr}} \land e > e_{\text{first\_fail}} > 0 \end{cases}$$

### 合成实验与定量验证（`experiments/mooncake_gid_reprobe_reshard_sim`）

运行 `python experiments/mooncake_gid_reprobe_reshard_sim/run.py --check` 对双栈 RoCEv2 GID 故障切换轨迹与 64 GiB（256 批 × 256 MiB）Scatter 重分片窗口扩展性进行确定性验证（数据源自 `experiments/mooncake_gid_reprobe_reshard_sim/results/summary.json`）：

| `max_inflight_batches` ($W$) | 观测峰值在途批次 | 完成批次数 | 64 GiB 总耗时 (`total_reshard_time_s`) | 有效数据面吞吐 (`effective_goodput_gbps`) | Pinned 窗口占用 (`pinned_window_footprint_mib`) | 未知 CQE 时状态机转换 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `1`（同步逐批基线） | `1` | `256` | `1.7562 s` | `313.03 Gbps` | `256 MiB` | `POISONED` |
| `2` | `2` | `256` | `1.6600 s` | `331.18 Gbps` | `512 MiB` | `POISONED` |
| **`4`（推荐甜点）** | **`4`** | **`256`** | **`1.5363 s`** | **`357.85 Gbps` (`+14.3%`)** | **`1,024 MiB`** | **`POISONED`** |
| `8` | `8` | `256` | `1.5363 s` | `357.85 Gbps` | `2,048 MiB` | `POISONED` |
| `16` | `16` | `256` | `1.5363 s` | `357.85 Gbps` | `4,096 MiB` | `POISONED` |

- **Auto-GID 状态机验证**：初始选中 GID index `3`（Overlay `172.17.0.2`）后，端点 `101` 首次故障触发 `kRetrySameGid`；端点 `101` 的重复报错被 `kIgnoreDuplicate` 拦截；重建端点 `102` 再次故障触发 `kAdvanceGid`，自动切换至 GID index `2`（`kNetworkPrivateV4`，`10.24.1.10`），并在端点 `103` 成功后将 `remaining_advances` 重置回满额 `2`。

### 架构师设计练习与可证伪问题

1. **为什么 `PeerAutoGidStateTable::recordCompletion` 必须要求 `attempt.endpoint_id > state.first_failed_endpoint_id` 才推进 GID，而不能简单统计失败 CQE 次数 $\ge 2$？**
   - **解答**：单次 RDMA 建连的端点（`RdmaEndPoint`）内部包含多个 QP 或深度为 `max_wr_depth` 的并发 WQE。当链路断开或超时时，同一个 `endpoint_id` 会在几微秒内连续刷出数十个错误 CQE。若按失败 CQE 计数，同一次瞬态抖动就会瞬间耗尽全部 `max_advances` 预算并把状态机打入 `kExhausted`。只有严格更大的 `endpoint_id` 才证明“上层已经销毁了旧端点并重建了第二个全新连接，但依然失败”。
2. **可证伪假设**：在 `RdmaEndPoint::~RdmaEndPoint()` 析构时，如果 `ibv_destroy_qp` 或 `ibv_dereg_mr` 返回非零错误码，能否照常释放 `notify_.send_buffer` 和 `notify_.recv_buffer` 主机内存？
   - **验证方式**：查阅 `rdma_endpoint.cpp:105-112`——当 `deconstructLocked()` 失败时，代码显式调用 `notify_.send_buffer.release()` 和 `notify_.recv_buffer.release()` 故意“泄漏”这两块 DMA 缓冲区而不将其归还给操作系统堆分配器，因为 RDMA 网卡硬件在销毁失败后仍可能向该物理地址执行 DMA 写入，释放该内存会导致严重的进程静默内存破坏（Silent Memory Corruption）。
