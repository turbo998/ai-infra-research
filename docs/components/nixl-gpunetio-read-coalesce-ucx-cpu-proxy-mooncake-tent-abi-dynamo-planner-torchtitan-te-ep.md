# NIXL GPUNetIO 连续 READ 跨块合并与 UCX CPU Proxy 专家并行、Mooncake Store `TentRequest` 48 字节 C ABI 与快照 Btime 淘汰、Dynamo Planner 热更新预热，以及 TorchTitan / TransformerEngine JAX EP 深挖

> **研究阶段**: Phase 46 (`priority: 55`) · **问题编号**: `q-phase46-nixl-gpunetio-read-ucx-proxy-mooncake-tent-dynamo-planner-torchtitan-te` · **确定性复现脚本**: [`experiments/nixl_gpunetio_read_ucx_proxy_mooncake_tent_dynamo_planner_torchtitan_te/run.py`](../../experiments/nixl_gpunetio_read_ucx_proxy_mooncake_tent_dynamo_planner_torchtitan_te/run.py)

---

## 1. 核心架构问题与一手提交总览

在跨节点 RDMA KV 缓存传输（NIXL / Mooncake）、大规模 MoE 专家并行（EP）通信代理、在线 SLA 自适应规划器（Dynamo Planner）以及分布式训练/XLA 编译期缓冲区重定位（TorchTitan / TransformerEngine）中，上游仓库近期落地了六项关键系统级改造：

1. **NIXL GPUNetIO 细粒度 READ WQE 洪泛与 `512` 输入分块边界割裂（`nixl#2176`）**：在 `4,096 × 4 KiB`（共 `16 MiB`）VRAM READ 负载下，主线逐描述符生成 `4,096` 个 WQE 并切分为 `8` 个 `DOCA_XFER_REQ_SIZE = 512` 环槽（Ring Entries）；仅在 512 块内合并仍会留下 `8 × 2 MiB` WQE 与 `8` 个环槽。
2. **NIXL UCX 后端缺乏多线程 CPU Proxy 驱动的 GPU 专家并行（`nixl#2327`, `nixl#2328`）**：GPU 直接发起的 `ucp_put_nbx` / 原子操作在某些 NIC/拓扑上受限于设备侧通道竞争；需要将 `channel_count * max_peers` 个设备侧 SPSC 环均匀切分给多个 CPU Proxy 线程轮询转发。
3. **Mooncake Store Rust FFI `TentRequest` 结构体大小失配（`Mooncake#4732`）与 `std::sort` 实时 `statx` 破坏严格弱序（`Mooncake#4688`）**：C 头文件中 `mooncake_tent_request_t` 包含 `priority` 与 `transport_hint`（共 `48` 字节），而 Rust 绑定 `TentRequest` 漏掉这两个字段（仅 `40` 字节），导致批量请求数组从第 2 个元素起发生 8 字节错位读取；同时 `StorageBackend::Init` 在 `std::sort` 比较器内实时调用 `statx(STATX_BTIME)`，当文件在排序中途被并发删除时破坏严格弱序（Strict Weak Ordering）并引发未定义行为。
4. **Dynamo Planner 缺乏在线无损热更新与历史轨迹预热校验（`dynamo#16843`）**：修改 `ttft`、`itl`、负载预测器类型或预热轨迹原本必须重启 Planner；若直接在主 Tick 循环内重训 ARIMA/Prophet/Kalman 会阻塞扩缩容决策。
5. **TorchTitan 冻结主干 + `lm_head` LoRA 分块损失跳过反向传播（`torchtitan#5389`）、HF 游标首轮迭代前丢失恢复状态（`torchtitan#5501`）与 TransformerEngine JAX `lax.scan` 句柄缓存悬空（`TransformerEngine#3639`）**。

| 仓库与提交 | 日期 | 核心架构变更 | 实测性能 / 正确性收益 |
|---|---|---|---|
| `nixl@1b84929f8a` (`#2176`) | 2026-10-15 | `canCoalesceRead` 在 `512` 输入块内与跨块边界合并同 Key 连续 READ（上限 `8 MiB`） | `4096×4KiB` READ WQE 从 `4096 -> 2`（**`2048x`** 缩减），环槽 `8 -> 2`；双 H20 吞吐 **`10.97 -> 23.03 GB/s`（`+109.92%`）**，p99 **`1516 -> 726 us`（`-52.11%`）** |
| `nixl@3f6e1a2716` (`#2327`) & `fda3a883c0` (`#2328`) | 2026-10-19 | `nixl::ucx::proxyEngine` 支持多线程 CPU Proxy 轮询 GPU SPSC 环并驱动 `nixl_ep` | 8 通道 / 8 线程下，16×H100（`45.77/42.19 GB/s`）与 16×GB200（`82.97/80.23 GB/s`）达到 **100%+** GPU 直连 RDMA 带宽 |
| `Mooncake@7e4fc11ee0` (`#4732`) & `1a6d00d0f4` (`#4688`) | 2026-10-20 | Rust `TentRequest` 补齐 `priority` + `transport_hint`（`40 -> 48` 字节）；`FileUtil::SortByCreationTime` 预快照 `TimeKey` | 消除 Rust/C FFI 批量传输内存错位与并发删除文件时的 `std::sort` 崩溃 |
| `dynamo@a76f1efaed` (`#16843`) | 2026-10-20 | `/engine/planner` 在线热重配（HTTP 409 版本锁、`16 MiB` / `4096` 步异步预热、`install_prepared_predictors`） | 零停机热切换预测器与 SLA 目标，并保留实时 `kv_hit_rate` 与 Tick 节拍 |
| `torchtitan@9391c0110a` (`#5389`) & `c0eb136092` (`#5501`) | 2026-10-19 | `ChunkedLossWrapper` 检查 `head_has_trainable_params`；`_HuggingFaceCursorIterator` 缓存 `_pending_hf_state` | 修复冻结 Decoder 仅训 `lm_head` LoRA 时梯度全零 Bug，以及首轮读取前连续 Checkpoint 丢失游标状态 |
| `TransformerEngine@6ca7b5aa88` (`#3639`) | 2026-10-14 | JAX Hybrid EP 设 `NVTE_EP_HANDLE_CACHE_SIZE=0` 并在每次 `dispatch_v2`/`combine_v2` 调 `ncclEpImportHandle` | 根除 XLA `lax.scan` 移动 `handle_mem` 缓冲区引发的悬空 `ncclEpHandle_t` 崩溃 |

---

## 2. NIXL：GPUNetIO 跨块连续 READ 合并与 UCX 多线程 CPU Proxy

### 2.1 `canCoalesceRead` 与跨 `DOCA_XFER_REQ_SIZE = 512` 边界合并 (`#2176`)

在 `src/plugins/gpunetio/gpunetio_backend_aux.h` 与 `gpunetio_backend.cpp` 中，`canCoalesceRead(previous, next, max_read_size)` 严格验证六项不变式：
1. `previous.size <= max_read_size` 且 `next.size <= max_read_size - previous.size`（其中 `max_read_size = min(8 MiB, DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE)`）；
2. `previous.localAddr <= UINTPTR_MAX - previous.size` 且 `previous.remoteAddr <= UINTPTR_MAX - previous.size`（无符号地址加法防溢出）；
3. `previous.localAddr + previous.size == next.localAddr` 且 `previous.remoteAddr + previous.size == next.remoteAddr`（双端虚拟地址严格连续）；
4. `previous.localKey == next.localKey` 且 `previous.remoteKey == next.remoteKey`（RDMA `lkey` / `rkey` 一致）。

在 `prepXfer` 中，当 `chunk_desc_count >= DOCA_XFER_REQ_SIZE`（512 个输入描述符）时，若 `operation == NIXL_READ` 且当前描述符可继续并入上一个已暂存 READ（`mergeable == true`），循环继续吸收后续描述符直至达到 `8 MiB` 上限；一旦遇到首个不可合并描述符（`!mergeable`），立即 `break` 关闭当前环槽且**不消耗**该描述符。对于 `4,096 × 4 KiB`（`16 MiB`）连续 VRAM READ：
- **Main 基线**：`4,096` 个 WQE，`8` 个环槽，双 H20 中位吞吐 **`10.97 GB/s`**，p99 **`1516 us`**；
- **Commit A（仅 512 块内合并）**：`8 × 2 MiB` WQE，`8` 个环槽，中位吞吐 **`20.39 GB/s`**，p99 **`810 us`**；
- **Commit B（块内 + 跨 512 边界合并）**：**`2 × 8 MiB` WQE，`2` 个环槽**（WQE 缩减 **`2048x`**），中位吞吐跃升至 **`23.03 GB/s`（较 Main `+109.92%`，较 Commit A `+12.97%`）**，p99 降至 **`726 us`（较 Main `-52.11%`）**。

### 2.2 UCX CPU Proxy `threadShare(thread)` 环分区与 EP 吞吐对比 (`#2327`, `#2328`)

`nixl::ucx::proxyEngine` 分配 `ringCount() = channel_count * max_peers` 个设备侧 SPSC 请求环，并通过 `evenThreadSplit()` 将 `(channel, peer)` 二维网格无重叠分配给 `thread_count` 个 CPU 轮询线程：
- 当 `thread_count <= channel_count` 时，每个线程独占 `channel_count / thread_count` 个完整通道（覆盖全部 Peer）；
- 当 `thread_count > channel_count` 时，每个通道由 `per_channel = thread_count / channel_count` 个线程按步长 `proxyPeerSlice{thread % per_channel, per_channel}` 分片轮询 Peer。

在 `nixl_ep` 专家并行基准（`#2328`）中，8 通道 / 8 线程 CPU Proxy 与 GPU 直连 RDMA（8 通道）对比如下：
- **16 GPUs (`2×8`) H100**：CPU Proxy Dispatch **`45.77 GB/s`** / Combine **`42.19 GB/s`** vs GPU RDMA `44.32 / 41.32 GB/s`（**103.3% / 102.1%**）；
- **16 GPUs (`4×4`) GB200**：CPU Proxy Dispatch **`82.97 GB/s`** / Combine **`80.23 GB/s`** vs GPU RDMA `81.90 / 76.62 GB/s`（**101.3% / 104.7%**）；
- **8 GPUs (`2×4`) GB300**：CPU Proxy Dispatch **`80.93 GB/s`** / Combine **`73.76 GB/s`** vs GPU RDMA `91.61 / 70.93 GB/s`（**88.3% / 104.0%**）。

---

## 3. Mooncake Store C ABI / 快照排序、Dynamo Planner 热重配与 TorchTitan / TE JAX

1. **Mooncake `TentRequest` 48 字节 C ABI 对齐 (`#4732`) 与 `SortByCreationTime` (`#4688`)**：在 `mooncake-store-rs/src/tent.rs` 中为 `TentRequest` 补齐 `pub priority: c_int`（偏移 `40`）与 `pub transport_hint: c_int`（偏移 `44`），使 `sizeof(TentRequest)` 从 `40` 字节恢复为与 C 端 `mooncake_tent_request_t` 完全一致的 **`48` 字节**；在 `FileUtil::SortByCreationTime` 中先将每个文件的 `statx(STATX_BTIME)` 快照为不可变 `TimeKey{!has_birth_time, seconds, nanoseconds}` 再调用 `std::sort`，保证严格弱序并将缺失创建时间的文件排在末尾。
2. **Dynamo Planner 在线热重配与离线循环外预热 (`#16843`)**：通过 `/engine/planner`（`GET` / `PATCH`）暴露带 `expected_revision` 乐观并发控制（冲突返回 HTTP `409`）的运行时重配接口；当预测器配置变更时，在 `asyncio.to_thread` 中对上限 `16 MiB`、`4096` 个时间步的历史轨迹执行 `prepare_predictors(..., strict=True)`（校验预测值有限且非负），随后在单把 `_reconfig_lock` 下通过 `install_prepared_predictors` 原子替换 `_num_req_predictor` / `_isl_predictor` / `_osl_predictor` 并保留实时 `kv_hit_rate`。
3. **TorchTitan `ChunkedLossWrapper` (`#5389`) 与 TransformerEngine JAX `ncclEpImportHandle` (`#3639`)**：`ChunkedLossWrapper` 将反向判定修正为 `run_backward = decoder_requires_grad or (torch.is_grad_enabled() and head_has_trainable_params)`，使冻结 Decoder + 可训练 `lm_head` LoRA 在无激活梯度时仍能跨 Chunk 累加参数梯度；TransformerEngine JAX 在 `extensions/jax/csrc/custom_call/ep.cu` 中默认设置 `NVTE_EP_HANDLE_CACHE_SIZE=0`，当缓存容量为 0 时不在主机 Map 中缓存 `ncclEpHandle_t`，而是在每次 `nvte_ep_dispatch_v2` / `nvte_ep_combine_v2` 中直接从 XLA 管理的 `handle_mem` 调用 `ncclEpImportHandle` 重建句柄并按序销毁。
