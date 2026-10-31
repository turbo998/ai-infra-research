# FlashInfer Cake LM-Head Step-Peak Budget & Valid Rows, SM120 NVFP4 Varlen Overlap, KDA N32 Bins, MXFP8 Prefetch, GB300 FMHA & Triton GSan (Phase 151)

> **Evidence Classification**: Primary source implementation audit (`implementation`) + deterministic stdlib offline simulation (`own-offline`). No hardware re-benchmarking is claimed.

## 1. 10 Primary Commit Audit Matrix (Zero-Overlap)

| # | Repository | Commit (`SHA`) | Date | Core Engineering Breakthrough & Code Path |
| :--- | :--- | :--- | :--- | :--- |
| 1 | `flashinfer-ai/flashinfer` | [`3292e01`](https://github.com/flashinfer-ai/flashinfer/commit/3292e01ba10ffbd78e5917bf85d5bd375dd09e9b) | 2026-10-29 | `feat(cake_lm_head_loss)` (`#7865`): Opt-in `step_peak_budget` and caller `valid_rows` count in `flashinfer/chunked_lm_head.py`. |
| 2 | `flashinfer-ai/flashinfer` | [`cf14a96`](https://github.com/flashinfer-ai/flashinfer/commit/cf14a9696ca3dc43929ed3da6923f2536bc73654) | 2026-10-29 | `perf(cake_minimax_h3)` (`#7854`): Plan-dependent overlap width for SM120 NVFP4 packed-varlen attention. |
| 3 | `flashinfer-ai/flashinfer` | [`710f9b5`](https://github.com/flashinfer-ai/flashinfer/commit/710f9b5a694545d4a65cc94c878ff35b1faaa525) | 2026-10-29 | `perf(cake_kda)` (`#7849`): Short bounded direct N32 launches take plain work bins when per-task binning cost dominates. |
| 4 | `flashinfer-ai/flashinfer` | [`47442df`](https://github.com/flashinfer-ai/flashinfer/commit/47442df411955c390261df6668018bd1b03c1b52) | 2026-10-29 | `perf(cake_grouped_mxfp8_quantize)` (`#7848`): Head-window L2 prefetch for two row-2D problem shapes in `128x4` scale layout. |
| 5 | `flashinfer-ai/flashinfer` | [`f319fc1`](https://github.com/flashinfer-ai/flashinfer/commit/f319fc1659a4a841450ca69dc3193f7ab17a7622) | 2026-10-29 | `perf(cake_fmha)` (`#7839`): GB300 FP8 context programs keep `P` in the other stage's `S` columns; B200 smaller-pack split plans. |
| 6 | `triton-lang/triton` | [`6caee94`](https://github.com/triton-lang/triton/commit/6caee941d0f13c01e4933443536108914f551a89) | 2026-10-22 | `[GSan] Add memory access profiler` (`#12338`): Adds `gsan-profile` instrumentation to suggest shadow granularity and `write_once` buffers. |
| 7 | `triton-lang/triton` | [`473737f`](https://github.com/triton-lang/triton/commit/473737fcb35a9c1836ef920df23a98944432a999) | 2026-10-21 | `[Runtime] Serialize class-valued constexprs for preload` (`#12430`): Serializes class objects by module and qualified name for JIT cache hooks. |
| 8 | `tile-ai/tilelang` | [`942bfde`](https://github.com/tile-ai/tilelang/commit/942bfdee0a8b8c9d7d478556f36f72c38f861d09) | 2026-10-08 | `[BugFix][CUDA] Support uint8 sparse GEMM operands` (`#3297`): Enables `uint8` sparse MMA macro generation. |
| 9 | `tile-ai/tilelang` | [`29b4fdd`](https://github.com/tile-ai/tilelang/commit/29b4fdd9544340ee55ee0993e5d7e81759a01b95) | 2026-10-08 | `[CUDA] Emit packed FP16 and BF16 vector comparisons` (`#3303`): Emits packed `half2`/`nv_bfloat162` comparison intrinsics. |
| 10 | `tile-ai/tilelang` | [`c819a83`](https://github.com/tile-ai/tilelang/commit/c819a835a33c9d176a52cfcfab9c7da13726c29d) | 2026-10-08 | `[BugFix][ROCm] Fix ties-to-even rounding for T.round` (`#3299`): Fixes IEEE-754 ties-to-even rounding on ROCm HIP. |

## 2. Deterministic Offline Verification (`own-offline`)

Run `python experiments/flashinfer_lmhead_minimax_kda_mxfp8_fmha_phase151/run.py --check`:
- **Cake LM-Head Compute Saving**: **`37.5%`** (`4,869.31 -> 3,043.32 GFLOPs`, peak memory `1216.5 -> 512.0 MB`)
- **SM120 NVFP4 Varlen Overlap Speedup**: **`1.2912x`**; **Cake KDA N32 Plain Bins Speedup**: **`1.3122x`**
- **Grouped MXFP8 Head-Window Prefetch Speedup**: **`1.2842x`**; **GB300 FMHA TMEM Reuse**: **`1.1964x`** (`256 -> 192` cols)
