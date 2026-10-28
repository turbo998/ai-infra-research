# FlashInfer MNNVL mHC Fused AllReduce, GB300 4-Stage FP8 Context FMHA Ring, CuTe DSL 5-Stage Page-Table Ring & Triton AMD Packed FP8 Upcast (Phase 130)

Deterministic CPU/offline analytical simulator covering 10 first-hand upstream commits in `flashinfer-ai/flashinfer` and `triton-lang/triton`:

1. `flashinfer-ai/flashinfer@1af7b11` (`#7758`): Eight-warp CTA programs (`dense_proj_gemm_kn_n256_tma1_w8rs` / `w8r`) on `sm_103a` for `4096->512` (`1.040x`), `5120->512` (`1.050x`), and `kv_a 6144->576` (`1.048x`).
2. `flashinfer-ai/flashinfer@cb91a3e` (`#7752`): 4-stage FP8 K/V ring on GB300 (`sm_103a`) achieving `1.081x` (host scale) and `1.078x` (device scale) speedup over 3-stage.
3. `flashinfer-ai/flashinfer@4c5d951` (`#7756`): Plan-dependent overlap width `P in {32, 48, 64}` for SM120 FP8 packed-varlen attention on 170-SM RTX 5090 and 188-SM RTX PRO 6000.
4. `flashinfer-ai/flashinfer@3c7e075` (`#6299`): Fixes CuTe DSL paged GQA decode page-table ring race by expanding `pt_stages` from `4` to `5` (`>= prefetch_iters + 3`), eliminating `s + 4` V-tile corruption (`30-50%` error).
5. `flashinfer-ai/flashinfer@3957698` (`#7695`): Ships all 30 Cake grouped FP8 GEMM programs on both `sm_100a` and `sm_103a` with `UE8M0` swap-AB small-block routing.
6. `flashinfer-ai/flashinfer@ae13e9c` (`#7682`): Hopper (`sm_90a`) VSA compact GQA two-step shuffle-exchange output stores cutting `SHFL` instructions from `97 -> 65` (BF16) and `90 -> 58` (Sage-FP8).
7. `flashinfer-ai/flashinfer@499012e` (`#7694`): `cake_muse_spark_norm` (`68 -> 77` configs) and `cake_tp_moe_finalize_rmsnorm` (`20 -> 22` configs) across B200/GB300/R200.
8. `flashinfer-ai/flashinfer@866313f` (`#6797`): Experimental `flashinfer.comm.allreduce_mhc_post_prenorm` fusing TP8 MNNVL one-shot all-reduce with 4-stream mHC post-map and 24-logit pre-norm partials across 89 sublayer boundaries in GLM-5.3-Flash decode (`267 -> 178` launches/step).
9. `triton-lang/triton@d3229a1` (`#12518`): Reverts premature single-warp partition `ttg.warp_id -> 0` folding before `NVGPUToLLVM`.
10. `triton-lang/triton@7e0b1a0` (`#12492`): Replaces per-element integer exponent add + 8-entry subnormal LUT in AMD `fp8e4nv -> fp16` software upcast with packed `i32` masking and `2^(15-7)` `fmul` (`up to 3.0x` RDNA3 GEMM speedup).

## Usage

```bash
python experiments/flashinfer_mhc_allreduce_fmha_ring_gqa_pt_triton_fp8_phase130/run.py --check
```
