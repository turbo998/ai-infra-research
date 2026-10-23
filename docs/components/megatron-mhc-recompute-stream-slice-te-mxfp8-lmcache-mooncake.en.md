# Deep Dive into Megatron-LM mHC/Wide-Residual Recompute Planning & Streaming Eviction Slicing, TransformerEngine Register-Resident MXFP8 (`exp2f_rcp_2x` Codes 254/255 & Swizzled Scales), TorchTitan Sticky Session `repin_load_ratio=1.5`, and LMCache/Mooncake/NIXL/Dynamo Transport Guards

> **Language / 语言**: [English](megatron-mhc-recompute-stream-slice-te-mxfp8-lmcache-mooncake.en.md) | [中文](megatron-mhc-recompute-stream-slice-te-mxfp8-lmcache-mooncake.md)  
> **Phase**: 70 (`priority: 79`, Track C — Distributed Training, RL Rollout Routing & KV/RDMA Storage Plane)  
> **Evidence Tiers**: `implementation` (fixed-commit source audit) + `author-benchmark` (author-measured TE Blackwell MXFP8 and TorchTitan RL CPU simulation reports) + `own-offline` (deterministic standard-library simulation `experiments/megatron-mhc-recompute-stream-slice-te-mxfp8-lmcache-mooncake/run.py`)

---

## 1. Architectural Context & Core Engineering Questions

Recent commits across distributed training stacks (Megatron-LM, TransformerEngine, TorchTitan) and KV/RDMA serving planes (LMCache, Mooncake, NIXL, Dynamo) resolve six critical correctness and performance bottlenecks:

1. **Megatron-LM Unified mHC & Wide-Residual Recompute Boundary Planning (`NVIDIA/Megatron-LM@c13b6e74e86d`, `#7923`) and Streaming Activation Eviction Segment Slicing (`@a54dd7378c25`, `#8150`)**: Why does placing a recompute checkpoint boundary inside an indivisible `atomic_layer_pairs` pair (e.g., an `(Attention, MLP)` wide-residual pair) break backward replay, and why did merged segments spanning an eviction checkpoint boundary desynchronize `_try_send_streaming_partials`?
2. **TransformerEngine Register-Resident MXFP8 Rowwise/Bidimensional Cast-Only Kernels & E8M0 `254/255` Fixes (`NVIDIA/TransformerEngine@9233c73a1b6e`, `#3459`)**: How do 256-bit `ld.global.nc.v8.b32` loads eliminate TMA staging and shared-memory payload traffic? Why did `exp2f_rcp_2x` (`0x7F007F00 - (biased_exp << 7)`) produce `0x0000` for E8M0 code `254` ($2^{-127}$ subnormal reciprocal) and borrow across BF16 lanes for code `255` (`NaN`), and how are 512B (`128x4`) GEMM-swizzled scales emitted across 4 warps without staging the payload?
3. **TorchTitan RL Sticky Session `repin_load_ratio=1.5` Straggler Eviction (`pytorch/torchtitan@50474ad6037c`, `#5219`) and Verifiers 0.4.0 Timeout Isolation (`@4103c7291ffe`, `#5456`)**: How does re-pinning a session when its generator's in-flight load exceeds `1.5x` the fleet mean recover **`8.0%–13.7%`** wall time under a `2x`-slow straggler while preserving prefix KV cache hits on balanced fleets, and why must `VerifiersRollouter.rollout_status` check `is_timeout` before `ok` on Verifiers 0.4.0?
4. **LMCache NIXL L2 Adapter Slot-Unit Capacity Reporting (`LMCache/LMCache@46e33e85a56d`, `#5800`) and InProcess Protocol/ZMQ Consolidation (`@10dd1e6a124d`, `#5979`)**: Why did `report_status()` return `pool_free_slots > pool_size` for multi-page POSIX files (`pages_per_file > 1`)?
5. **Mooncake TENT `io_uring` Short-Completion Rejection (`kvcache-ai/Mooncake@4012f2a56f69`, `#4618`) and `USE_ASCEND_DIRECT` `:npu_x` Suffix Stripping (`@d04eaf621bce`, `#4627`)**: Why must nonnegative `io_uring` CQEs with `cqe->res < request.length` copy only `cqe->res` bytes and fail the task, and why did `USE_ASCEND_DIRECT` trigger `EADDRINUSE` (`503900`) during init?
6. **NIXL Two-Process UCX Distributed Trace Correlation (`ai-dynamo/nixl@13a896bde498`, `#2404`) and Dynamo `/health?instances=true` Opt-In Serialization & TRT-LLM Fatal Exit (`ai-dynamo/dynamo@a789babbe37a` `#16454`, `@2ae8a46299b1` `#16633`)**: Why did default `/health` probes spend tens of milliseconds serializing discovery instances on large deployments, and why must Dynamo skip `engine.shutdown()` when the TRT-LLM executor already recorded a fatal error?

---

## 2. Core Implementation Paths & Source-Level Mechanics

### 2.1 Megatron-LM `build_recompute_block_end_plan` & `_slice_merged_segments` (`#7923`, `#8150`, `#7926`)

1. **Unified mHC & Wide-Residual Recompute Plan (`#7923`)**:
   - `build_recompute_block_end_plan(num_layers, recompute_method, recompute_num_layers, atomic_layer_pairs)` in `megatron/core/transformer/transformer_block.py` computes a boolean per-layer end-of-block plan shared by `_checkpointed_forward` and mHC recompute signaling:
     - Under `uniform`, a block closes at `layer_idx` when `(layer_idx - block_start + 1) >= recompute_num_layers` **and** `(layer_idx, layer_idx + 1) not in atomic_layer_pairs` (or at the final layer), extending blocks by 1 layer whenever cutting at `layer_idx` would split an atomic wide-residual `(Attention, MLP)` pair.
2. **Eviction-Aligned Segment Slicing (`#8150`) & Removing `disable_ep_consensus` (`#7926`)**:
   - In `megatron/core/pipeline_parallel/eviction_schedules.py`, `_slice_merged_segments` splits merged consecutive layer segments at every activation-eviction checkpoint boundary so `_try_send_streaming_partials` fires at the exact segment boundaries expected by the backward schedule; `#7926` removes `disable_ep_consensus`.

### 2.2 TransformerEngine Register-Resident MXFP8 Cast-Only Kernels & E8M0 Fixes (`#3459`)

In `transformer_engine/common/cast/mxfp8/specialized/{cast_rowwise,cast_bidim}.cuh` and `util/ptx.cuh`, TE adds header-only register-resident BF16 MXFP8 cast-only kernels gated on `sm_arch() >= 100` and 32-byte input pointer alignment:

- **Register-Resident Streaming & Closed-Form E8M0 Math**:
  - **Rowwise (`cast_rowwise.cuh`)**: Two lanes cooperate per 32-element MX block (each lane issues one 256-bit `ld.global.nc.v8.b32` for 16 BF16 elements and one 128-bit store for 16 FP8 elements). Folding the rowwise remainder loop via `CHECK_BOUNDS` cuts `8160x1920` latency from `10.66 -> 8.96 us` (**`-15.9%`**) and `4064x4480` from `11.10 -> 9.47 us` (**`-14.7%`**).
  - **Bidimensional (`cast_bidim.cuh`)**: Each 8-warp (256-thread) CTA covers a 32-row band (matching the 32-element colwise block height), reading its tile into registers **once** to drive both rowwise (warp shuffle) and colwise (shared-memory amax) passes without TMA staging—improving geomean execution time vs the TMA kernel from `0.957` to **`0.864`**.
  - **Closed-Form E8M0 Derivation**: For a finite BF16 magnitude $\text{amax} = m \cdot 2^{e-127}$, $\lceil \log_2(\text{amax}/448)\rceil + 127$ equals $e - 8$ for $m \le 1.75$ (7-bit mantissa $\le 96$) and $e - 7$ for $m > 1.75$ (mantissa $\ge 97$). Adding `31 << 7` carries into bit 7 of the exponent iff mantissa $\ge 97$, avoiding FP32 `1/448` rounding errors.
- **`exp2f_rcp_2x` and `mx_scale_byte_x2` Fixes for Codes `254` (`Inf`) and `255` (`NaN`)**:
  - `exp2f_rcp_2x(biased_exp)` computes `0x7F007F00u - biased_exp * 0x00800080u`, which is valid only for normal BF16 reciprocals (`biased_exp <= 253`). For code `254` ($2^{-127}$ subnormal, `0x0040`), the subtraction yielded `0x0000` (scaling `Inf` blocks by zero!); for code `255` (`NaN`), the low lane borrowed out of the high lane, producing asymmetric garbage `0xFF7FFF80`.
  - **Fix**: `if (__builtin_expect(biased_exp >= 254, 0))` selects `0x00400040u` for `254` and `0x7FFF7FFFu` for `255`; `cast_bidim.cuh` separates `kInfReciprocal` (`254`) from `kNaNReciprocal` (`255`), gives each half of `mx_scale_byte_x2` carry headroom, and tests `(magnitude + 0x0080)` bit 15 so `max.xorsign.abs` cannot swallow a paired NaN column.
- **4-Warp 512B GEMM-Swizzled Scale Tile**:
  - To emit the `128 rows x 4 scale cols` (512B) GEMM-swizzled scale layout without staging payload in SMEM, `cast_rowwise.cuh` spreads the 4 sub-rows (32 rows apart) across 4 warps of a 256-thread CTA; each warp reads a contiguous row run, and only the scale bytes meet in a few hundred bytes of SMEM to flush as 16-byte stores.

### 2.3 TorchTitan Sticky Session `repin_load_ratio=1.5` (`#5219`) & Verifiers 0.4.0 (`#5456`)

1. **`StickySessionRoutingStrategy` `repin_load_ratio=1.5` (`#5219`)**:
   - Before reusing a session's pinned candidate $c_{\text{pin}}$, `choose()` computes $\bar{L} = \frac{1}{N}\sum_h L(h)$ and checks $L(c_{\text{pin}}) > 1.5 \bar{L}$. When overloaded (e.g., `[6, 2, 2, 2]` with mean `3.0` $\implies 6 > 4.5$), the fallback selects the least-loaded candidate and re-pins the session there, cutting multi-turn wall time by **`8.0%`** (`13.7%–15.2%` when KV-bound) under a `2x`-slow straggler in CPU simulations.
2. **Verifiers 0.4.0 Timeout Isolation (`#5456`)**:
   - Because Verifiers 0.4.0 can leave `ok=True` when a stage times out while setting `trace.is_timeout=True`, `VerifiersRollouter.rollout_status` checks `if trace.is_timeout or verifiers_episode.is_timeout: return RolloutStatus.ERROR_TIMEOUT` first and gates reward assignment on `if not status.is_error():`.

### 2.4 LMCache NIXL Slot Reporting (`#5800`/`#5979`), Mooncake `io_uring` Short-I/O Guards (`#4618`/`#4627`), and NIXL/Dynamo (`#2404`/`#16454`/`#16633`)

- **LMCache NIXL Pool Capacity (`#5800`) & InProcess Move (`#5979`)**: `NixlStoreL2Adapter.report_status()` now reports `"pool_size": pool.total_objs` (slot units = files $\times$ `pages_per_file`) rather than `self._config.pool_size` (file count); `#5979` moves `RemoteMetadata` and `IPC_SOCKET_PATH_MAX_LEN = 107` into `lmcache.inprocess.v1`.
- **Mooncake TENT `io_uring` Short Completions (`#4618`) & `USE_ASCEND_DIRECT` (`#4627`)**: `IOUringTransport::getTransferStatus` copies only `completed_bytes = static_cast<size_t>(cqe->res)` on staged reads and marks `status_word = (completed_bytes == request.length) ? COMPLETED : FAILED`; `#4627` strips `:npu_x` under `USE_ASCEND_DIRECT` to prevent `EADDRINUSE` (`503900`).
- **NIXL Cross-Process UCX Trace (`#2404`) & Dynamo `/health` / TRT-LLM Fatal Exit (`#16454`, `#16633`)**: NIXL verifies shared correlation IDs across `ucx_trace_receiver` via `libtrace_backend_recorder.so`; Dynamo `/health` returns `{"status": "healthy", "stage": stage}` by default and serializes discovered instances only on `/health?instances=true`; Dynamo's TRT-LLM monitor stops admission (`runtime.shutdown()`) first, skips `engine.shutdown()` when `get_health_check_fatal_error()` is set, bounds non-fatal teardown on a daemon thread, and exits via `os._exit(1)`.

---

## 3. Deterministic Offline Verification (`own-offline`)

Experiment script: `experiments/megatron-mhc-recompute-stream-slice-te-mxfp8-lmcache-mooncake/run.py` (results saved in `summary.json`):

| Verification Target | Parameters | Measured / Simulated Output | Architectural Takeaway |
|---|---|---|---|
| Megatron `build_recompute_block_end_plan` | 6 layers, `uniform K=2`, `atomic_layer_pairs={(1,2)}` | Without atomic: `[F,T,F,T,F,T]`; with `(1,2)`: `[F,F,T,F,T,T]` | Prevents recompute checkpoints from splitting wide-residual/mHC atomic pairs |
| TE MXFP8 `exp2f_rcp_2x` boundary codes | `biased_exp = 254` (Inf), `255` (NaN) | Legacy `0x00000000` / `0xFF7FFF80` $\to$ Fixed `0x00400040` / `0x7FFF7FFF` | Restores exact $2^{-127}$ subnormal reciprocal and prevents cross-lane borrow |
| TorchTitan `repin_load_ratio=1.5` & Verifiers | Loads `[6,2,2,2]` vs `[4,3,3,2]`; `is_timeout=True, ok=True` | `[6,2,2,2]` moves to `idx=1`; `[4,3,3,2]` stays `idx=0`; status `ERROR_TIMEOUT` | Evicts stragglers without thrashing prefix caches or training timed-out rollouts |
| LMCache NIXL & Mooncake `io_uring` | `16` files $\times 4$ pages/file; `req=4096B, cqe=1024B` | `pool_size=64 >= free_slots=62`; short read `FAILED`, unread tail preserved | Fixes capacity inversion and prevents silent partial-read corruption |

---

## 4. Solutions Architect Exercises & Falsifiable Questions

1. **Derivation Exercise**: In TransformerEngine `cast_bidim.cuh`, why does adding `31 << 7` to a BF16 bit pattern carry into the exponent bit 7 if and only if the mantissa $m > 1.75$?
   - *Answer*: With implicit leading 1, the 7-bit mantissa field $f \in [0, 127]$ represents $m = 1 + f/128$. Since $1.75 = 1 + 96/128$, $m > 1.75 \iff f \ge 97 \iff f + 31 \ge 128 = 2^7$, which carries 1 into exponent bit 7.
2. **Falsifiable Hypothesis**: Should Kubernetes readiness probes on a large Dynamo deployment query `/health?instances=true` every second?
   - *Falsification*: `/health?instances=true` serializes every discovered worker endpoint and instance on the async runtime (`10+ ms` per probe at scale); readiness probes should query default `/health` and reserve `?instances=true` for topology diagnostics.
