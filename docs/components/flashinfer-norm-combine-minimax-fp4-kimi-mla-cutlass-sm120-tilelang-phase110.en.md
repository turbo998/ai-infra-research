# FlashInfer Fused Norm-Combine, MiniMax-H3 SM120 Pre-Attention GEMM, NVFP4 L2 Prefetch, Kimi-K3 TP12 Persistent Reduce & MLA, MoE AR Union, CUTLASS & TileLang (Phase 110)

> 中文版: [flashinfer-norm-combine-minimax-fp4-kimi-mla-cutlass-sm120-tilelang-phase110.md](flashinfer-norm-combine-minimax-fp4-kimi-mla-cutlass-sm120-tilelang-phase110.md) | Companion experiment: `experiments/flashinfer_norm_combine_minimax_fp4_kimi_mla_phase110/run.py`

## 1. First-Hand Code Evidence (2026-08-06 to 2026-10-27)

| Repo | Commit | Date | Core Mechanism Change | Evidence Class |
|---|---|---|---|---|
| `flashinfer` | `3da22ba71f` | 2026-10-27 | `cake_fused_norm_combine` streamlines the 38..479-token stream prologue (SMEM `128 -> 256 B` with `8 B` handoff) and reads the workspace table earlier in the 1-token program (`1.026x-1.034x` B200 speedup) | `implementation` + `author-benchmark` |
| `flashinfer` | `5b2aacd211` | 2026-10-27 | `cake_minimax_h3` optimizes `sm_120a` FP8/NVFP4 pre-attention GEMM epilogues across 6/9 programs (up to `1.0798x` on RTX PRO 6000 and `1.0469x` on RTX 5090) | `implementation` + `author-benchmark` |
| `flashinfer` | `e158799310` | 2026-10-27 | `cake_mm_fp4` enables 4-stage entry L2 weight prefetch on 3 more cells (`sm_100a` `n8_aF_l2256b_o2` at `M=1..8`, `sm_103a` `m64_bF_l2256b_ro` at `M=128`, and 8 standalone GEMM rows at `M=128`) | `implementation` + `author-benchmark` |
| `flashinfer` | `531e0bc2ba` | 2026-10-27 | `cake_kimi_k3_tp12_tail` introduces scatter-free persistent reduce mode `k3_reduce:grouped` for `320 <= M <= 960` with 2-token in-flight gathers (`1.056x` on GB300 at `M=960`) | `implementation` + `author-benchmark` |
| `flashinfer` | `45411ad7d7` | 2026-10-27 | `cake_kimi_k3_mla` adds late-trigger `16/32/48`-row tiles (`main CTAs >= 32`) and pre-wait split-KV reducer setup (`1.1649x` on B200 and `1.1295x` on GB300 for Shape 1) | `implementation` + `author-benchmark` |
| `flashinfer` | `acb2bef7ec` | 2026-10-27 | `cake_trtllm_moe_allreduce_union` runs 1-token rows at `2 CTAs/SM` (`LAUNCH_MIN_BLOCKS=2`) with `st.async` RMS exchange (`1.03x-1.13x` faster across all 30 shapes on B200/B300) | `implementation` + `author-benchmark` |
| `cutlass` | `976d0521f0` | 2026-08-06 | `cute::operator==(ScaledBasis, ScaledBasis)` guards value comparison behind matching bases, unblocking SM120 `256x128x256` block-scaled sparse MMA and asserting `TileShape_K` divisibility | `implementation` |
| `cutlass` | `3c063071bd` | 2026-08-06 | Fixes `AffineRank2RowMajor::packed(extent)` stride order to `(extent.column(), 1)` instead of column-major `(1, extent.row())` | `implementation` |
| `tilelang` | `cb6455d680` | 2026-10-19 | Fixes shared-memory `T.cumsum` under warp specialization by passing `thread_offset = thread_bounds->min` and checking `threadIdx.x - thread_offset >= SEG` | `implementation` |
| `tilelang` | `9253431f2a` | 2026-10-19 | Adds ROCm HIP codegen support for `float4_e2m1fn` constants via `__tl_float_to_fp4(...)` and sets `enable_fp4_` in `VisitExpr_(FloatImmNode)` to emit `hip_fp4.h` | `implementation` |

## 2. Key Implementation & Architectural Takeaways

1. **PDL Entry Prefetch & Late-Trigger Split-KV Reduction**: `cake_mm_fp4` prefetches the first `4` K-stages of weights into L2 before the PDL grid wait, while `cake_kimi_k3_mla` delays split-KV reducer launch on plans with `>= 32` main CTAs and overlaps reducer row setup before waiting, achieving `1.1649x` speedup on B200 at batch-1 decode.
2. **Warp-Specialized Thread Offsets & Packed Stride Invariants**: Subtracting `thread_offset = 128` in TileLang's `InclusiveScan1D/2D` restores all `32` active scan lanes on consumer warps (versus `0` active lanes previously), and CUTLASS `AffineRank2RowMajor::packed(64, 128)` corrects linear offset at `(3, 5)` from `323` to `389`.
