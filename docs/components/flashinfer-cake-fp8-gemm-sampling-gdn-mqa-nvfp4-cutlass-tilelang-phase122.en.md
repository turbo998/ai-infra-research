# FlashInfer CAKE Grouped FP8 GEMM 32-Row UE8M0 Swap-AB, Paged Auto Q-Len Refresh, CAKE Sampling Round 16 SM120, CAKE GDN Bwd, CAKE DeepGEMM Dense MQA, NVFP4 TMA Row Bound, CUTLASS 4.8 SM107/MXFP8 & TileLang Autotuner (Phase 122)

> **中文版**: [flashinfer-cake-fp8-gemm-sampling-gdn-mqa-nvfp4-cutlass-tilelang-phase122.md](flashinfer-cake-fp8-gemm-sampling-gdn-mqa-nvfp4-cutlass-tilelang-phase122.md)
> **Phase**: Phase 122 (`priority: 131`) · Track A Kernel & Quantization Micro-Architecture
> **Experiment Script**: [`experiments/flashinfer_cake_fp8_gemm_sampling_gdn_mqa_nvfp4_cutlass_tilelang_phase122/run.py`](../../experiments/flashinfer_cake_fp8_gemm_sampling_gdn_mqa_nvfp4_cutlass_tilelang_phase122/run.py)

---

## 1. Primary Sources & Evidence Classification

| Evidence / Source ID | Repo & PR/Commit | Merge Date | Classification | Core Technical Contribution |
|---|---|---|---|---|
| `src-flashinfer-cake-grouped-fp8-gemm-ue8m0-swap-ab-small-block-sm103-pr7589-2026` | [`flashinfer-ai/flashinfer#7589`](https://github.com/flashinfer-ai/flashinfer/pull/7589) (`87658682`) | 2026-10-28 | `implementation` + `author-benchmark` | Adds two SM103 (`sm_103a`) 32-row M-block UE8M0 swap-AB programs (`bs_ue8m0_swap_rp32_s11` with 11 stages & 2-CTA clusters; `bs_ue8m0_swap_c1` with 1 CTA per `32x128` tile), reaching **1.72x** speedup (`13.63 -> 7.93 us`) on GB300 (`152` SMs) |
| `src-flashinfer-paged-auto-refresh-retained-q-len-metadata-fa2-fallback-pr7617-2026` | [`flashinfer-ai/flashinfer#7617`](https://github.com/flashinfer-ai/flashinfer/pull/7617) (`06577b40`) | 2026-10-28 | `implementation` | Refreshes retained `_cudnn_seq_lens_q` metadata when `BatchPrefillWithPagedKVCacheWrapper(backend="auto")` falls back to FA2 with caller-supplied `block_tables`, eliminating **67.4%** output mismatch and NaNs on SM107 (`48/48` cells pass) |
| `src-flashinfer-cake-sampling-round16-sm120-c8-e40-radix-topk-pr7604-2026` | [`flashinfer-ai/flashinfer#7604`](https://github.com/flashinfer-ai/flashinfer/pull/7604) (`9657c04e`) | 2026-10-27 | `implementation` + `author-benchmark` | `cake_sampling` Round 16 adds `(8, 40)` stage-1 radix top-k kernels (`106 -> 108` builds) for RTX PRO 6000 (`sm_120a`, `188` SMs, `101,376 B` dynamic SMEM), achieving **92.7%** register residency at `V=151,936` without spills and up to **1.151x** speedup |
| `src-flashinfer-cake-gdn-train-bwd-swizzled-dqkg-tape-prefetch-sm107-pr7602-2026` | [`flashinfer-ai/flashinfer#7602`](https://github.com/flashinfer-ai/flashinfer/pull/7602) (`9c81aefd`) | 2026-10-27 | `implementation` + `author-benchmark` | `cake_gdn_train` backward `dqkg` XOR lane swizzle + 128B packed stores, 1-step-ahead `2 x 256-bit` `dv_intra` tape prefetch, and `ctas_per_sm=2` (`296` CTAs) for `bwd_prep` on SM107 (`25.283 -> 24.733 ms`, **1.022x** on GB300 P1) |
| `src-flashinfer-cake-deepgemm-dense-mqa-warp-reduction-sm100-sm103-pr7607-2026` | [`flashinfer-ai/flashinfer#7607`](https://github.com/flashinfer-ai/flashinfer/pull/7607) (`5073cd43`) | 2026-10-27 | `implementation` + `author-benchmark` | Replaces 15 dense MQA logits programs on `sm_100a`/`sm_103a` with warp-wide cost-boundary scan (`768` blocks/round), fused `f32x2` abs+add/FMA, and dropped pre-sync `tcgen05` fence (**1.0537x** median / **1.1199x** max on B200; **1.0458x** median / **1.1012x** max on GB300) |
| `src-flashinfer-nvfp4-quantize-tma-scale-factor-row-bound-pr6278-2026` | [`flashinfer-ai/flashinfer#6278`](https://github.com/flashinfer-ai/flashinfer/pull/6278) (`e9c4787c`) | 2026-10-27 | `implementation` | Guards NVFP4 TMA quantize scale-factor padding writes with `threadRowIdxGlobal < numPaddedRowsForSf`, preventing `n / 2` bytes (`1024 B` at `n=2048`) out-of-bounds writes on `layout_8x4` when `m % 16 in [1, 8]` |
| `src-cutlass-v48-rubin-sm107-mxfp8-k64-sf-k128-tile-pr3645-2026` | [`NVIDIA/cutlass#3645`](https://github.com/NVIDIA/cutlass/pull/3645) (`098de2a6`) | 2026-09-22 | `implementation` | CUTLASS 4.8.0 introduces Rubin (`SM107`) CuTe DSL extensions (`MMA_K=64` FP8, `MMA_K=128` FP4, `576` TMEM cols, `328 KB` SMEM) and `is_mxfp8_k64_sf_k128_tile` sharing one `K=128` SF chunk across two `K=64` iterations |
| `src-tilelang-autotuner-get-tunner-forward-profile-args-pr3337-2026` | [`tile-ai/tilelang#3337`](https://github.com/tile-ai/tilelang/pull/3337) (`90f8908b`) | 2026-10-19 | `implementation` | Forwards `warmup, rep, timeout` in `AutoTuneImpl.get_tunner()` to `set_profile_args()`, eliminating cache-key collisions (`hash=74468074`) across distinct tuning budgets |

---

## 2. Core Architectural Analysis & Derivations

### 2.1 CAKE Grouped FP8 GEMM 32-Row M-Block UE8M0 Swap-AB (`#7589`)

On Blackwell Ultra (`sm_103a`, GB300 `152` SMs), 1D1D UE8M0 block-scaled grouped FP8 GEMMs with moderate token counts `M` and narrow output channels `N` suffer from severe expert tail-padding waste under 64-row or 128-row M-tiles. PR `#7589` adds two 32-row swap-AB programs (swapping activation `A` and weight `B` operands in the tensor core MMA so the 32-row dimension maps to M):

1. **`bs_ue8m0_swap_rp32_s11` (11 stages, 2-CTA cluster)**:
   $$\text{CTAs}_{\text{rp32\_s11}}(M, N) = 2 \cdot \left\lceil \frac{M}{32} \right\rceil \cdot \frac{N}{256} \le \text{SM\_count}, \quad K \ge 2048$$
   At the boundary shape `M=2432, N=256, K=4096`, $\text{CTAs} = 2 \cdot 76 \cdot 1 = 152$, fitting GB300 (`152` SMs) in a single wave while falling back on `148`-SM devices. On `b3_a64_n256_k4096_p16` (`M=2048, routed=1417, N=256, K=4096`), latency drops from `13.63 -> 7.93 us` (**1.72x**).
2. **`bs_ue8m0_swap_c1` (1 CTA per `32x128` tile, no cluster)**:
   $$\text{CTAs}_{\text{c1}}(M, N) = \left\lceil \frac{M}{32} \right\rceil \cdot \frac{N}{128} \le \text{SM\_count}, \quad \text{align}=32, \; 256 \le K \le 512, \; N \le 512$$

### 2.2 Paged `backend="auto"` Retained Q-Length Refresh (`#7617`) & NVFP4 TMA Row Bound (`#6278`)

- **Paged Auto Retained Q-Length Refresh (`#7617`)**: When the caller passes an external `block_tables` tensor on the first `plan()`, `self._cudnn_block_tables` remains `None` while `self._cudnn_seq_lens_q` caches the first plan's Q lengths. If a second `plan()` falls back to FA2 on ragged Q, guarding the refresh on `self._cudnn_block_tables is not None` left `_cudnn_seq_lens_q` stale; when a third `plan()` returned to cuDNN, it read stale Q lengths and caused **67.4%** output mismatches and NaNs on SM107. Guarding on `self._cudnn_seq_lens_q is not None` resolves all `48/48` test cells.
- **NVFP4 TMA Scale-Factor Row Bound (`#6278`)**: `quantize_with_block_size_tma` launches `TMA_ROW_TILE = 16` rows for FP16/BF16, whereas `SFMatrixLayout::layout_8x4` pads scale-factor allocation to multiples of `8` rows (`numPaddedRowsForSf = ceil(m / 8) * 8`). When `m % 16 in [1, 8]` (e.g., `m = 1025`), `1040` rows execute against `1032` allocated SF rows, writing $8 \times (n / 16) = n / 2$ bytes (`1024 B` at `n=2048`) out of bounds. Adding `threadRowIdxGlobal < numPaddedRowsForSf` eliminates the OOB write and removes Python-side `round_up(m, 16)` over-allocation.

### 2.3 CAKE Sampling Round 16 SM120 (`#7604`), CAKE GDN Bwd (`#7602`), Dense MQA (`#7607`), CUTLASS 4.8 (`#3645`) & TileLang (`#3337`)

- **CAKE Sampling Round 16 (`#7604`)**: Adds `(cluster=8, ept=40, threads=512)` stage-1 radix top-k kernels (`capacity = 163,840`, `dynamic_smem_bytes = 35,968` within SM120's `101,376 B` opt-in limit), raising Qwen `V=151,936` register utilization from `77.3%` (`ept=48`, spilling) to **92.7%** without spills (**1.151x** speedup on RTX PRO 6000 `188` SMs).
- **CUTLASS 4.8 SM107 & MXFP8 (`#3645`) & TileLang (`#3337`)**: Adds Rubin `SM107` (`576` TMEM columns, `328 KB` SMEM) and `is_mxfp8_k64_sf_k128_tile` (sharing one `K=128` SF tile across two `K=64` MXFP8 iterations), while TileLang fixes `AutoTuneImpl.get_tunner()` to forward `(warmup, rep, timeout)` into `set_profile_args()`.
