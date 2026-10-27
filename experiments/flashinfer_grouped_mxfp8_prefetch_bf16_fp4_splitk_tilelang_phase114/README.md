# FlashInfer Grouped MXFP8 Prefetch, Blackwell BF16×FP4 Split-K, MiniMax-H3 Norm Stage, CUTLASS SM75 & TileLang Ascend SS/PTO (Phase 114)

Deterministic CPU/offline verification suite for Phase 114 Track A (`priority: 123`):
- `flashinfer-ai/flashinfer@f23e264378`: Per-K row-tiled `cake_grouped_mxfp8_quantize` programs prefetching partial-group rows 2–3 groups ahead (`80.95%` exposed stall reduction).
- `flashinfer-ai/flashinfer@88063fd695`: Small-`M` split-K `cake_blackwell_bf16_fp4` programs on SM100/SM103 with `allow_oob_box = {false, true, true}`.
- `flashinfer-ai/flashinfer@9b08bf90a3`, `c1e28c1e16`, `a50ec035cd`, `d8ce22c0fe`: `BLOCK_N=48` 8-stage `SMEM_EPI_STAGING_STAGE_BYTES=12288` (`SMEM_TOTAL=201728`), `NORM_V5` shared-memory staging for `P=8, M>=6096` on `sm_103a`, and 3-stage `seq % 3` mbarrier addressing in Mega-MHC.
- `NVIDIA/cutlass@6c68991985`, `80fe3cac71` & `tile-ai/tilelang@7617c7d94d`, `c55eeb1c02`: CuTe `SM75_16x8x8_F16F16F16F16_TN`, `smem_ptr_flag_bits` header fix, Ascend `shared.ss` 3072-byte 32B-aligned scalar buffer pool (`kMix`, `mix_aiv_count_==2`), and `execution_backend="pto"` codegen routing.

```bash
python experiments/flashinfer_grouped_mxfp8_prefetch_bf16_fp4_splitk_tilelang_phase114/run.py --check
```
