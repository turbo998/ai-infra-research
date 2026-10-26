# FlashInfer Cake QSA NVFP4, MiniMax-H3 SM103 4-Dest Norm, Grouped MXFP8 Early Exit, FP8 Attention Folded K-Mean Bias, GDP Tail-Wave & TileLang Bijection/Analyzer Guards (Phase 96)

> **Evidence Classification**: `implementation` (pinned upstream commits) + `own-offline` (deterministic Python 3.12 standard-library verification; not GPU hardware profiling).

## 1. Primary Upstream Source Evidence (10 Zero-Overlap Commits)

| Source ID | Repo / Commit | Key Implementation Path & Architectural Mechanism |
|---|---|---|
| `flashinfer-cake-qsa-nvfp4-early-blocktable-partial-epilogue-pr7460-2026` | `flashinfer-ai/flashinfer` `e400000d1c` (PR #7460) | Hoists paged KV block-table loads ahead of the TMA prologue in `cake_qsa_nvfp4` `tcgen05` decode on SM100/SM103 and writes split-KV partial outputs directly in the epilogue (`direct_epilogue_smem_roundtrips = 0`). |
| `flashinfer-cake-minimax-h3-sm103-4dest-nvfp4-norm-pr7458-2026` | `flashinfer-ai/flashinfer` `2751459afe` (PR #7458) | Routes `cake_minimax_h3` pre-attention normalization with `P == 4` destinations on `sm_103a` (B300) to a dedicated 4-destination NVFP4 `NORM` program alongside `FUSED_PACK_K96`/`FUSED_E4M3_K96` GEMMs. |
| `flashinfer-cake-grouped-mxfp8-early-exit-prefetch-partial-pr7457-2026` | `flashinfer-ai/flashinfer` `32f46c223f` (PR #7457) | Per-K row-tiled `cake_grouped_mxfp8_quantize` programs immediately exit CTAs whose starting row exceeds `valid_rows` and prefetch partial expert group bounds. |
| `flashinfer-cake-fp8-attention-fold-kmean-bias-exp2-pr7429-2026` | `flashinfer-ai/flashinfer` `050e27087d` (PR #7429) | Folds the smoothed-K mean bias $q \cdot \mu_K$ directly into the `exp2` softmax scale-bias argument and unrolls `stats_finalize` by `unroll = 8`. |
| `flashinfer-cake-trtllm-mla-blackwell-aligned-dense-decode-pr7427-2026` | `flashinfer-ai/flashinfer` `1769ac7474` (PR #7427) | Specializes aligned dense Multi-head Latent Attention (`MLA`) decode schedules on `sm_100a` and `sm_103a`. |
| `flashinfer-cake-gdp-tail-wave-band-gate-logit-carrier-pr7393-2026` | `flashinfer-ai/flashinfer` `69e33dae7b` (PR #7393) | Introduces the `cake_gdp` `T = 1` tail-wave band rule ($4 \cdot \text{SM} < B \cdot HV \le 5 \cdot \text{SM} \to \text{one\_row\_per\_thread\_l8r4}$), `T = 4` gate-logit carrier, and SM100 `T = 8` dense-load carrier. |
| `flashinfer-cake-gated-act-mxfp8-packed-f32x2-sigmoid-pr7396-2026` | `flashinfer-ai/flashinfer` `93e5508a15` (PR #7396) | Switches 18 per-architecture `cake_gated_act_mxfp8` programs (`4` on `sm_100a`, `9` on `sm_103a`, `5` on `sm_107a`) to packed `f32x2` sigmoid evaluation with `128x4` swizzled E8M0 `RCEIL` scales. |
| `tilelang-revert-nan-propagating-clamp-lowering-pr3628-2026` | `tile-ai/tilelang` `a2cf4c2b1e` (PR #3628) | Reverts `T.clamp(x, lo, hi)` lowering to `T.min(T.max(x, lo), hi)` across CPU/CUDA codegen. |
| `tilelang-reject-non-bijective-sm8x-sparse-metadata-pr3125-2026` | `tile-ai/tilelang` `ced0ca1a30` (PR #3125) | Enforces `M % group == 0` (`group = 32` for 16-bit, `16` for 8-bit) and `K % 2 == 0` in `make_cutlass_metadata_layout_sm8x` to guarantee bijective 2:4 sparse metadata layouts. |
| `tilelang-avoid-stale-analyzer-bindings-loop-partition-pr3226-2026` | `tile-ai/tilelang` `741bf8b722` (PR #3226) | Passes `const arith::Analyzer *analyzer` into `LowerParallelLoop` and `GetVectorizeSize` and snapshots into a local analyzer so partitioned loop bounds never leak stale bindings. |

## 2. Mathematical Modeling & Architect Exercises

1. **Folded K-Mean Bias Identity & Per-K Row-Tiled Early CTA Exit**:
   Folding $q \cdot \mu_K$ into the base-2 exponent achieves $< 10^{-14}$ numerical difference while removing separate post-loop bias rescaling. For `valid_rows = 384`, `max_padded_rows = 1024`, and `tile_m = 128`, exiting CTAs past `valid_rows` retires `5` of `8` CTAs per K-slice early (`saved_cta_fraction = 0.625`).
2. **Cake GDP Tail-Wave Band Rule**:
   On a 148-SM Blackwell GPU (`sm_100a`), when $592 < B \cdot HV \le 740$ (e.g., $B \cdot HV = 656$), routing `T = 1` decode to `one_row_per_thread_l8r4` avoids the occupancy/register pressure of the 5-CTA/SM `l8r4w5` kernel.

## 3. Reproducible Offline Verification

```bash
python experiments/flashinfer_qsa_nvfp4_minimax_mxfp8_fp8_attn_mla_gdp_gated_act_tilelang_phase96/run.py --check
```
Verified in `summary.json`: `direct_epilogue_smem_roundtrips == 0`, `saved_cta_fraction == 0.625`, `total_packed_programs == 18`, `selected_gdp_t1_program == "one_row_per_thread_l8r4"`, and `sm8x_invalid_fp16_48x128 == False`.
