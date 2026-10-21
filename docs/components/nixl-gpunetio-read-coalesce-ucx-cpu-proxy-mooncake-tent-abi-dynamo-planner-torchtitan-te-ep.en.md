# Deep Dive: NIXL GPUNetIO Contiguous READ Coalescing & UCX CPU Proxy EP, Mooncake Store `TentRequest` 48-Byte C ABI & Snapshot Btime Eviction, Dynamo Planner Warmup Reconfiguration, and TorchTitan / TransformerEngine JAX EP

> **Research Phase**: Phase 46 (`priority: 55`) · **Question ID**: `q-phase46-nixl-gpunetio-read-ucx-proxy-mooncake-tent-dynamo-planner-torchtitan-te` · **Deterministic Script**: [`experiments/nixl_gpunetio_read_ucx_proxy_mooncake_tent_dynamo_planner_torchtitan_te/run.py`](../../experiments/nixl_gpunetio_read_ucx_proxy_mooncake_tent_dynamo_planner_torchtitan_te/run.py)

---

## 1. Executive Summary & First-Hand Upstream Commits

Across RDMA KV-cache transport (NIXL / Mooncake), expert-parallel (EP) communication engines, online SLA autoscaling planners (Dynamo), and distributed training / XLA buffer management (TorchTitan / TransformerEngine), recent upstream commits deliver six production systems advancements:

1. **NIXL GPUNetIO Contiguous READ Coalescing Across the `512`-Input Staging Boundary (`nixl#2176`)**: On a contiguous `4,096 × 4 KiB` (`16 MiB`) VRAM READ, unmodified `main` emits `4,096` WQEs across `8` ring entries (`DOCA_XFER_REQ_SIZE = 512`), while within-chunk coalescing still emits `8 × 2 MiB` WQEs across `8` ring entries.
2. **NIXL UCX Multi-Threaded CPU Proxy for GPU Expert Parallelism (`nixl#2327`, `nixl#2328`)**: Enables GPU kernels to post descriptors and fences into device-visible SPSC rings while multi-threaded CPU proxy workers drive UCX RDMA operations via `evenThreadSplit()` and `threadShare(thread)`.
3. **Mooncake Store Rust `TentRequest` 48-Byte C ABI Alignment (`Mooncake#4732`) & Snapshot `SortByCreationTime` (`Mooncake#4688`)**: Adds missing `priority` and `transport_hint` fields to Rust's `TentRequest` (`40 -> 48` bytes) to match C's `mooncake_tent_request_t`, and snapshots `statx(STATX_BTIME)` into `TimeKey` before `std::sort` so disappearing files cannot violate strict weak ordering.
4. **Dynamo Planner Live Reconfiguration & Off-Loop Predictor Warmup (`dynamo#16843`)**: Adds `/engine/planner` (`GET`/`PATCH`) with `expected_revision` optimistic locking (HTTP `409`), off-loop `prepare_predictors` warmup (`16 MiB` / `4096` intervals max, finite nonnegative validation), and atomic `install_prepared_predictors` swap.
5. **TorchTitan Frozen-Decoder `lm_head` LoRA Chunked Loss (`torchtitan#5389`), HF Cursor State Persistence (`torchtitan#5501`), and TransformerEngine JAX `lax.scan` EP Handle Reconstruction (`TransformerEngine#3639`)**.

| Commit | Date | Architectural Mechanism | Measured / Verified Impact |
|---|---|---|---|
| `nixl@1b84929f8a` (`#2176`) | 2026-10-15 | `canCoalesceRead` merges adjacent same-key READs within & across the 512-input boundary (`8 MiB` cap) | Collapses `4096×4KiB` READ from `4096 -> 2` WQEs (**`2048x`**) and `8 -> 2` ring entries; 2×H20 median **`10.97 -> 23.03 GB/s` (`+109.92%`)**, p99 **`1516 -> 726 us` (`-52.11%`)** |
| `nixl@3f6e1a2716` (`#2327`) & `fda3a883c0` (`#2328`) | 2026-10-19 | `nixl::ucx::proxyEngine` (`device_proxy=true`, `threadShare(thread)`) running `nixl_ep` over UCX | At 8ch/8thr, achieves **`45.77 / 42.19 GB/s`** on 16×H100 and **`82.97 / 80.23 GB/s`** on 16×GB200 (**100%+** of direct GPU RDMA) |
| `Mooncake@7e4fc11ee0` (`#4732`) & `1a6d00d0f4` (`#4688`) | 2026-10-20 | Rust `TentRequest` adds `priority` + `transport_hint` (`40 -> 48` bytes); `FileUtil::SortByCreationTime` snapshots `TimeKey` | Eliminates 8-byte/element FFI stride corruption on batch `Put`/`Get` and `std::sort` UB on concurrent file deletion |
| `dynamo@a76f1efaed` (`#16843`) | 2026-10-20 | `/engine/planner` runtime reconfiguration with `expected_revision`, off-loop warmup (`16 MiB` / `4096` steps), atomic swap | Zero-downtime SLA & predictor updates while preserving live `kv_hit_rate` and tick schedule |
| `torchtitan@9391c0110a` (`#5389`) & `c0eb136092` (`#5501`) | 2026-10-19 | `ChunkedLossWrapper` checks `head_has_trainable_params`; `_HuggingFaceCursorIterator` caches `_pending_hf_state` | Accumulates `lm_head` LoRA gradients across chunks when decoder is frozen; preserves HF cursor state across `set_state -> get_state` |
| `TransformerEngine@6ca7b5aa88` (`#3639`) | 2026-10-14 | Sets `NVTE_EP_HANDLE_CACHE_SIZE=0` for JAX and reconstructs `ncclEpHandle_t` via `ncclEpImportHandle` per call | Prevents dangling `ncclEpHandle_t` crashes when XLA `lax.scan` relocates `handle_mem` |

---

## 2. NIXL GPUNetIO Cross-Chunk READ Coalescing & UCX Multi-Threaded CPU Proxy

### 2.1 `canCoalesceRead` and Cross-`512`-Boundary Extension (`#2176`)

`canCoalesceRead(previous, next, max_read_size)` merges two READ ranges only when:
1. `previous.size <= max_read_size` and `next.size <= max_read_size - previous.size` (`max_read_size = min(8 MiB, DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE)`);
2. Neither `previous.localAddr + previous.size` nor `previous.remoteAddr + previous.size` wraps `uintptr_t`;
3. Both local and remote addresses are strictly contiguous and `localKey` / `remoteKey` match.

In `nixlDocaEngine::prepXfer`, once `chunk_desc_count >= DOCA_XFER_REQ_SIZE` (`512`), the loop continues consuming descriptors **only** while `mergeable == true` for the active READ run, breaking before incrementing `desc_offset` at the first non-mergeable descriptor. Across five 2×H20 benchmark pairs on `4,096 × 4 KiB` (`16 MiB`) VRAM READs:
- **Unmodified `main`**: `4,096` WQEs, `8` ring entries, **`10.972816 GB/s`** median throughput, **`1516 us`** p99 transfer latency;
- **Commit A (within-chunk)**: `8 × 2 MiB` WQEs, `8` ring entries, **`20.389650 GB/s`** median throughput, **`810 us`** p99;
- **Commit B (combined cross-chunk)**: **`2 × 8 MiB` WQEs, `2` ring entries** (`2,048x` WQE reduction), **`23.034552 GB/s`** median throughput (**`+109.92%` over `main`, `+12.97%` over A**), **`726 us`** p99 (**`-52.11%` over `main`**).

### 2.2 UCX CPU Proxy `threadShare(thread)` Ring Partitioning (`#2327`, `#2328`)

`nixl::ucx::proxyEngine` partitions `channel_count * max_peers` SPSC rings across `thread_count` CPU threads: when `thread_count <= channel_count`, each thread polls `channel_count / thread_count` full channels; when `thread_count > channel_count`, `per_channel = thread_count / channel_count` threads share each channel via peer-strided `proxyPeerSlice{thread % per_channel, per_channel}`. At 8 channels / 8 threads, the CPU proxy achieves **`45.77 / 42.19 GB/s`** (dispatch/combine) on 16×H100 and **`82.97 / 80.23 GB/s`** on 16×GB200, matching 100%+ of direct GPU RDMA.

---

## 3. Mooncake Store C ABI / Snapshot Sort, Dynamo Planner Warmup & TorchTitan / TE JAX

1. **Mooncake `TentRequest` 48-Byte C ABI (`#4732`) & Snapshot `SortByCreationTime` (`#4688`)**: Appending `priority: c_int` (offset `40`) and `transport_hint: c_int` (offset `44`) aligns Rust's `TentRequest` to C's **48-byte** `mooncake_tent_request_t` (fixing an 8-byte/element stride drift). `FileUtil::SortByCreationTime` snapshots `statx(STATX_BTIME)` into `TimeKey{!has_birth_time, seconds, nanoseconds}` prior to `std::sort`.
2. **Dynamo Planner Runtime Reconfiguration (`#16843`)**: Validates `expected_revision` (HTTP `409` on mismatch), warms up candidate predictors off the tick loop (`16 MiB` / `4096` intervals max, `strict=True`), and atomically swaps predictors under `_reconfig_lock`.
3. **TorchTitan `ChunkedLossWrapper` (`#5389`) & TransformerEngine JAX (`#3639`)**: `run_backward = decoder_requires_grad or (torch.is_grad_enabled() and head_has_trainable_params)` enables frozen-decoder `lm_head` LoRA training with chunked cross-entropy; `NVTE_EP_HANDLE_CACHE_SIZE=0` forces per-call `ncclEpImportHandle` reconstruction when XLA `lax.scan` relocates `handle_mem`.
