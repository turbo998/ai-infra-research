# FlashInfer Muse Spark RMSNorm Route Tuning, SM120 GDN Element-Pair Slot RW, SM100a 8-Warp Dense Projection GEMM & TileLang IR Caching (Phase 152)

> **Evidence Classification**: Primary source implementation audit (`implementation`) + deterministic stdlib offline simulation (`own-offline`).

## 1. 10 Primary Commit Audit Matrix (Zero-Overlap)

| # | Repository | Commit (`SHA`) | Date | Core Engineering Breakthrough & Code Path |
| :--- | :--- | :--- | :--- | :--- |
| 1 | `flashinfer-ai/flashinfer` | [`8f14bf2`](https://github.com/flashinfer-ai/flashinfer/commit/8f14bf2e792073cb755a57e508be77fd8bb9a087) | 2026-10-30 | `fix(cake_lm_head_loss)` (`#7916`): Validates caller `valid_rows` bounds on uncompacted budget calls. |
| 2 | `flashinfer-ai/flashinfer` | [`5cc23cd`](https://github.com/flashinfer-ai/flashinfer/commit/5cc23cd27585dc0efee78480501bf9d17b189444) | 2026-10-29 | `perf(cake_muse_spark_norm)` (`#7844`): Retunes gated residual RMSNorm on GB300 and next-RMSNorm on B200, GB300, and R200. |
| 3 | `flashinfer-ai/flashinfer` | [`19af158`](https://github.com/flashinfer-ai/flashinfer/commit/19af158bcdf1b5b93099237300413fb385116220) | 2026-10-29 | `perf(gdn)` (`#7837`): `sm_120` slot calls read and write state rows as vectorized element pairs. |
| 4 | `flashinfer-ai/flashinfer` | [`bd98569`](https://github.com/flashinfer-ai/flashinfer/commit/bd9856910efeb9b19cb6b96d51b8fc2948be35fa) | 2026-10-29 | `perf(cake_dense_projection_gemm)` (`#7835`): Eight-warp CTA program for SM100a input-gradient projection shapes. |
| 5 | `flashinfer-ai/flashinfer` | [`b83db54`](https://github.com/flashinfer-ai/flashinfer/commit/b83db54b4d2a16be1a23bf8cb860a02fa6554e9e) | 2026-10-29 | `perf(cake_sampling)` (`#7830`): GB300 and B200 host launch choices for two cells. |
| 6 | `triton-lang/triton` | [`3b0d4f8`](https://github.com/triton-lang/triton/commit/3b0d4f81cb7599a01b95010a6bd9f53d508190d0) | 2026-10-23 | `[Tools] Fix cached function and global alias source slicing` (`#12464`). |
| 7 | `triton-lang/triton` | [`0385e7a`](https://github.com/triton-lang/triton/commit/0385e7ae4566abfb71fa92cf76db35efcaef6f91) | 2026-10-22 | `[Frontend][triton_kernels] Remove reshape reordering opt-ins` (`#12457`). |
| 8 | `tile-ai/tilelang` | [`fcd97d4`](https://github.com/tile-ai/tilelang/commit/fcd97d4f62189ce2a54fb4c8b136c13726c89342) | 2026-10-14 | `[Test] Optimize testing infrastructure with cached IR traversal` (`#3502`). |
| 9 | `tile-ai/tilelang` | [`fbd7cf8`](https://github.com/tile-ai/tilelang/commit/fbd7cf8b86182c7d97268d21a1a949cf85b2c257) | 2026-10-08 | `[Bugfix][Quantize] Use plain int masks in interleave_weight special branches` (`#3140`). |
| 10 | `tile-ai/tilelang` | [`47976f2`](https://github.com/tile-ai/tilelang/commit/47976f2c7344fa707271110b2e1abbd525e842f8) | 2026-10-07 | `[BugFix][JIT] Preserve variadic arguments when building lazy kernels` (`#3435`). |

## 2. Deterministic Offline Verification (`own-offline`)

Run `python experiments/flashinfer_muse_norm_gdn_pairs_dproj_sampling_phase152/run.py --check`:
- **Muse Spark RMSNorm Speedup (B200/GB300/R200)**: **`1.3396x`** (`28.4 -> 21.2 us`)
- **SM120 GDN Element-Pair Slot RW Speedup**: **`1.3609x`** (`46.0 -> 33.8 us`)
- **SM100a 8-Warp Dense Projection GEMM Speedup**: **`1.3393x`** (`112.5 -> 84.0 us`)
- **TileLang Cached IR Traversal Speedup**: **`7.069x`** (`410.0 -> 58.0 ms`)
