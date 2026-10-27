# FlashInfer Grouped MXFP8 Multi-Group Prefetch, Blackwell BF16×FP4 Split-K OOB TMA, MiniMax-H3 Norm Shared Staging, CUTLASS SM75 & TileLang Ascend SS/PTO (Phase 114)

> **Evidence Classification**: `implementation` + `own-offline` (upstream commit analysis and reproducible CPU/offline model verification; not real GPU hardware profiling)

## 1. Upstream Code Paths & Microarchitectural Mechanics

| Evidence ID | Upstream Commit | Key Implementation Path & Microarchitectural Invariant |
|---|---|---|
| `flashinfer-cake-grouped-mxfp8-quantize-partial-group-prefetch-2026` | `flashinfer-ai/flashinfer@f23e264378` (`2026-10-27`) | Per-K row-tiled `cake_grouped_mxfp8_quantize` programs prefetch valid rows of a partial group **2–3 groups ahead**, cutting exposed partial-tile memory stalls by `80.95%` in our offline pipeline model. |
| `flashinfer-cake-blackwell-bf16-fp4-small-m-splitk-2026` | `flashinfer-ai/flashinfer@88063fd695` (`2026-10-27`) | Small-`M` split-K `cake_blackwell_bf16_fp4` programs on SM100/SM103 enforce per-axis TMA bounds via `allow_oob_box[3] = {false, true, true}` and streamline `smk_part` workspace indexing. |
| `flashinfer-cake-batch-deepgemm-fp8-blockn48-8stage-sm103a-2026` | `flashinfer-ai/flashinfer@9b08bf90a3` (`2026-10-27`) | Deploys an 8-stage `BLOCK_N=48` lean-epilogue program on `sm_103a` with `SMEM_EPI_STAGING_STAGE_BYTES = 12288` (`SMEM_TOTAL = 201728` bytes). |
| `flashinfer-cake-minimax-h3-nvfp4-norm-8dest-sm103-2026` | `flashinfer-ai/flashinfer@c1e28c1e16` (`2026-10-27`) | Routes `sm_103a` `P == 8, out_format="nvfp4", M >= 6096` (as well as `P == 4`) to `NORM_V5`, which stages row activations in shared memory with by-value `CUtensorMap` descriptors. |
| `flashinfer-cake-deepgemm-mega-mhc-sm107a-tail-2026` | `flashinfer-ai/flashinfer@a50ec035cd` (`2026-10-27`) | Adds `sm_107a` FP8/NVFP4 tail variants in `deepgemm_mega_mhc` and separates 3-stage mbarrier slot (`seq % 3`) from parity (`(seq / 3) % 2 ^ 1`). |
| `flashinfer-fp4-quantization-grouped-nvfp4-mask-scale-doc-2026` | `flashinfer-ai/flashinfer@d8ce22c0fe` (`2026-10-27`) | Documents `[B, M, 2 * K]` gate+up input layout and `int32 [B]` per-group valid row counts for fused SiLU-mul grouped NVFP4 quantization. |
| `cutlass-cute-sm75-16x8x8-f16-tn-mma-2026` | `NVIDIA/cutlass@6c68991985` (`2026-08-18`) | Adds `SM75_16x8x8_F16F16F16F16_TN` MMA atom and `MMA_Traits` for Turing Tensor Cores. |
| `cutlass-cute-print-tensor-smem-ptr-flag-bits-2026` | `NVIDIA/cutlass@80fe3cac71` (`2026-08-06`) | Includes `<cute/pointer_flagged.hpp>` in `include/cute/util/print_tensor.hpp` for `cute::smem_ptr_flag_bits`. |
| `tilelang-ascend-ss-buffer-support-2026` | `tile-ai/tilelang@7617c7d94d` (`2026-10-16`) | Adds `T.alloc_ss` (`shared.ss` scope, `volatile __ssbuf__`) sharing a 3072-byte 32B-aligned scalar pool across 1 AIC + 2 AIVs (`kMix && mix_aiv_count_ == 2`). |
| `tilelang-ascend-route-pto-execution-codegen-2026` | `tile-ai/tilelang@c55eeb1c02` (`2026-10-15`) | Routes `target="ascend", execution_backend="pto"` directly through `PTO_CODEGEN` and PTO kernel cache dispatch. |

## 2. Verification Command

```bash
python experiments/flashinfer_grouped_mxfp8_prefetch_bf16_fp4_splitk_tilelang_phase114/run.py --check
```
