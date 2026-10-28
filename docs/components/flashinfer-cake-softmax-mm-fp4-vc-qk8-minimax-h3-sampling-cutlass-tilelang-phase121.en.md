# FlashInfer CAKE Softmax Async-Proxy Fence, MM FP4 Plan Fallback, VC-Attention-QK8 Context FMHA, QSA NVFP4 Thread-Safe Plan Table, MiniMax H3 SM103 Shared-Memory Residual Staging, Autotuner Delay Stream Binding, Bounded Tensor Top-K Fast Path, Radix Top-K Static SMEM Budget, CUTLASS Block-Scale Epilogue Batch Indexing & TileLang Uncached Compile Testing (Phase 121)

> **Module Positioning**: Written for cloud and AI infrastructure solution architects mastering Blackwell (`sm_100a`), Blackwell Ultra (`sm_103a`), and GeForce Blackwell (`sm_120`) kernel memory-ordering contracts, quantized attention, and sampling microarchitecture. Based on 10 primary upstream commits landed in `flashinfer-ai/flashinfer`, `NVIDIA/cutlass`, and `tile-ai/tilelang` between 2026-07-18 and 2026-10-28.
> **Evidence Scope**: `implementation` (verified against pinned commit diffs) + `author-benchmark` (upstream PR measurements, not re-run on project GPUs) + `own-offline` (reproducible Python standard-library modeling in this repository).

---

## 1. Architectural Overview & 10 Upstream Commits

| # | Repository & Commit | Date | Core Mechanism & Architectural Change | Evidence Scope |
| :--- | :--- | :--- | :--- | :--- |
| A1 | `flashinfer-ai/flashinfer` [`fc4df7e`](https://github.com/flashinfer-ai/flashinfer/commit/fc4df7ee1a1edbd713ff1438c25f8de6b3e5d3af) | 2026-10-28 | `cake_softmax`: Insert `fence.proxy.async.shared::cta;` after `mbarrier_wait(freed_addr, 0)` before persistent softmax refills a freed tile slot | `implementation` + `author-benchmark` |
| A2 | `flashinfer-ai/flashinfer` [`dd21f60`](https://github.com/flashinfer-ai/flashinfer/commit/dd21f60a61303b4555412f3a3d11c06de5968a0b) | 2026-10-28 | `cake_mm_fp4`: Fall back to a registered and fitting program on `unregistered_key` or `unfit_key` | `implementation` |
| A3 | `flashinfer-ai/flashinfer` [`e91141b`](https://github.com/flashinfer-ai/flashinfer/commit/e91141bddada2d373a879bdea7c19b5fdeff0b77) | 2026-10-28 | `prims-ts`: `VC-attention-QK8` for context FMHA (Centered K + 128-point Hadamard-rotated E4M3 Q/K with block scales) | `implementation` + `author-benchmark` |
| A4 | `flashinfer-ai/flashinfer` [`015fff1`](https://github.com/flashinfer-ai/flashinfer/commit/015fff182331b5a11621e7699dac1055602c1d09) | 2026-10-28 | `cake_qsa_nvfp4`: Thread-safe plan-table operations (`pop(ident, None)` and exception-tolerant `_evict_oldest`) | `implementation` |
| A5 | `flashinfer-ai/flashinfer` [`056670d`](https://github.com/flashinfer-ai/flashinfer/commit/056670d018638b536169c6691f548576bf3c464e) | 2026-10-27 | `cake_minimax_h3`: Stage residual in shared memory (`228,480 B` SMEM, `waves >= 8`) for SM103 (`sm_103a`) NVFP4 tail-split out-projection | `implementation` + `author-benchmark` |
| A6 | `flashinfer-ai/flashinfer` [`93ab5d9`](https://github.com/flashinfer-ai/flashinfer/commit/93ab5d91038eb79a11c97dec81b7c5e31730105a) | 2026-10-28 | `autotuner`: Wrap scalar `delay_kernel()` FFI call with `with tvm_ffi.use_torch_stream():` | `implementation` + `author-benchmark` |
| A7 | `flashinfer-ai/flashinfer` [`ba25ff7`](https://github.com/flashinfer-ai/flashinfer/commit/ba25ff79fdfab20af333f132c9c70a96c50f9dfa) | 2026-10-27 | `sampling`: Add `top_k_max` bound enabling zero-host-sync top-k-first fast path for per-row tensor `top_k` | `implementation` + `author-benchmark` |
| A8 | `flashinfer-ai/flashinfer` [`205c51e`](https://github.com/flashinfer-ai/flashinfer/commit/205c51eaa90db9b12c9d386e50ed6b68fe4bfa) | 2026-10-27 | `sampling`: Reserve static shared memory (`attr.sharedSizeBytes = 336 B`) in `GetRadixTopKLaunchConfig` | `implementation` + `author-benchmark` |
| A9 | `NVIDIA/cutlass` [`ad9bd53`](https://github.com/NVIDIA/cutlass/commit/ad9bd53bdaec27a2e88053d57322ccf74efe525e) | 2026-07-18 | `sm100_block_scale_otto`: Save `int l_norm_const = tile_coord_l` before zeroing `tile_coord_l` in SM100/SM120 block-scale epilogue stores | `implementation` |
| A10 | `tile-ai/tilelang` [`0408712`](https://github.com/tile-ai/tilelang/commit/0408712ead01b4383e0f63e9e0bf74c35c1d7676) | 2026-10-28 | `tilelang/testing`: Add `without_kernel_cache()` context manager preserving caller's `CacheState` | `implementation` |

---

## 2. Deep Technical Analysis & Formulations

### 2.1 Async-Proxy Fence in Persistent Row-Caching Cluster Softmax (`fc4df7e`)

In `c8_t8` and `c16_t8` persistent row-caching softmax programs (`flashinfer/cake/kernels/softmax/program.py`):
- Compute warps write normalized outputs to shared-memory tile slots via the generic proxy (`st.shared`) and release `freed_addr` via `cp.async.bulk` completion. When Warp 0 wakes from `mbarrier_wait(freed_addr, 0)` and immediately refills the freed slot via `cp.async.bulk.global.shared::cta`, the PTX memory consistency model requires an explicit async-proxy fence (`fence.proxy.async.shared::cta;`) between generic-proxy shared-memory accesses and subsequent async-proxy bulk refills.
- Adding `fence.proxy.async.shared::cta;` eliminates intermittent data races across **4,000 back-to-back launches (0 errors)** on both B200 (`sm_100a`) and GB300 (`sm_103a`) with a **0.996x** 51-shape geomean ratio (`author-benchmark`).

### 2.2 VC-Attention-QK8 Context FMHA & SM103 MiniMax H3 Shared-Memory Residual Staging (`e91141b`, `056670d`)

- **VC-Attention-QK8 (`e91141b`)**: Centers $K$ ($\tilde{K} = K - \mu_K$), applies a 128-point Fast Walsh-Hadamard Transform (FWHT) to $Q$ and $\tilde{K}$, and quantizes both into block-scaled FP8 E4M3 (`q_block_size` up to `256`, `128`-token K tile). By folding $\text{sf}_Q \cdot \text{scale} \cdot \log_2 e$ into a per-row scale and prefetching $\text{sf}_K$, latency on B200 for Wan2.2 P1 (`B=2, H=40, S=75.6k`) drops from `166.0 ms` (BF16 dense) and `142.0 / 134.0 ms` (`VC-QK16`) down to **`123.0 ms` (means, 1.3496x) and `115.0 ms` (repair, 1.4435x)** (`author-benchmark`).
- **SM103 MiniMax H3 Shared-Memory Residual Staging (`056670d`)**: Exploits GB300's `228,480 B` shared memory per SM to allocate two `2 KiB` residual staging slots per epilogue warp when $\text{waves} \ge 8$ ($\ge 608$ tile pairs across 76 CTA pairs), achieving a **1.010x** 15-shape geomean speedup (`P=4, M=8368`: `125.2 -> 123.5 us`, **1.0138x**).

### 2.3 Bounded Tensor `top_k_max` Fast Path & SM120 Radix Top-K Static SMEM Budget (`ba25ff7`, `205c51e`, `93ab5d9`)

- **Bounded Tensor `top_k_max` (`ba25ff7`)**: Enables `_top_k_first_fast_path` for 1D per-row `top_k` tensors without calling `top_k.max()` (which would synchronize the host), selecting `top_k_max` sorted candidates first and masking each row to `top_k[row]`. On B200 (`vocab=262144, batch=256, top_k=20, top_p=0.9`), latency drops from `0.819 ms` to **`0.218 ms` (3.7569x speedup)**.
- **SM120 Radix Top-K Static SMEM Budget (`205c51e`)**: Queries `cudaFuncGetAttributes` (`attr.sharedSizeBytes = 336 B`) so `GetRadixTopKLaunchConfig` rejects `num_Pass2_threads = 1024` ($101,312 + 336 = 101,648 > 101,376$ B on RTX 5090 SM120) and selects `512` threads ($99,264 + 336 = 99,600 \le 101,376$ B), restoring **558/558** test passes.
- **Autotuner Delay Stream Binding (`93ab5d9`)**: Wraps scalar `delay_kernel()` in `with tvm_ffi.use_torch_stream():`, increasing captured non-default-stream delay on H100 NVL from `0.019 ms` to **`5.147 ms`**.

---

## 3. Reproducible Offline Verification (`own-offline`)

Experiment directory: `experiments/flashinfer_cake_softmax_mm_fp4_vc_qk8_minimax_h3_sampling_cutlass_tilelang_phase121/`

- **Command**: `python experiments/flashinfer_cake_softmax_mm_fp4_vc_qk8_minimax_h3_sampling_cutlass_tilelang_phase121/run.py --check`
- **Verified Sealed Outputs**:
  - `cake_softmax_fence.stress_errors_after_fix = 0`, `b200_51shape_geomean_vs_main = 0.996`;
  - `vc_qk8_and_minimax_h3.wan22_p1_vc_qk8_repair_speedup = 1.4435`, `minimax_h3_sm103_min_tile_pairs = 608`;
  - `autotuner_and_sampling.max_sampling_speedup = 3.7569`, `threads_1024_total_smem_bytes = 101648`, `threads_512_total_smem_bytes = 99600`;
  - `cutlass_and_tilelang.batch_indexing_restored = true`.

---

## 4. Architect Exercises & Falsifiable Questions

1. **Calculation Exercise**: On an SM120 GPU (`max_smem_per_block = 101,376 B`), if a kernel reports `attr.sharedSizeBytes = 1,024 B`, what is the maximum dynamic shared memory (`smem_size`) the launch config may request?
   - *Answer*: $101,376 - 1,024 = 100,352\text{ Bytes}$.
2. **Falsifiable Question**: When `top_k_top_p_sampling_from_probs` runs on pre-normalized flat distributions (`rand()/sum`) at `vocab >= 151,936` and `batch >= 64`, does the two-stage `top_k_max` path incur higher memory-traffic overhead than single-pass full-vocab masking?
