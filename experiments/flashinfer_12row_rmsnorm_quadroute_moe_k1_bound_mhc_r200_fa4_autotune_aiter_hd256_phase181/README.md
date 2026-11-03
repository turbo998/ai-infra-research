# Phase 181: GPU Software Stack Deterministic Verification Suite

Validates the quantitative kernel & compiler claims for Phase 181 (`priority: 190`):
- **FlashInfer `#8505` (`1798b527fa18`)**: `cake_rmsnorm_silu_gate_train` 12-rows/SM short forward on B200 (`1.312x`, up to `1,776` rows) and GB300 (`1.304x`, up to `1,824` rows).
- **FlashInfer `#8503` (`283300b79da4`)**: `cake_moe_finalize_allreduce_fusion` quad-route WS2 programs for `tokens <= 142` (`1.070x` max on B300 at `T=128`).
- **FlashInfer `#8456` (`27b38c15c3a7`)**: `cake_dense_projection_gemm` K1 bound argument lists (`1.684x` `2fwd+bwd` speedup on GB300).
- **FlashInfer `#8488` (`dc4bf3420d01`) & `#8449` (`936cd32a539a`)**: `cake_mhc_det` dual-layout `fn` split on R200 `sm_107a` (`1.033x`) and two-wave low-latency Blackwell softmax (`1.096x`).
- **CuTe FlashAttention `#2743` (`54b044c8cf89`) & `#3091` (`1dc8232fe209`)**: Subprocess barrier warm autotuning on GB300 (`11.0x`, `4.4 s -> 0.4 s`) and MLA CTA kernel choices (`3`).
- **Triton (`#12572`, `#12601`, `#12522`), TileLang (`#3689`), and ROCm AITER (`#7173`, `#5578`)**: `gfx950` `hd256` MXFP4 `mha_v4` reaching `3,652 TFLOP/s` (`1.185x` vs `hd128`).

## Run
```bash
python experiments/flashinfer_12row_rmsnorm_quadroute_moe_k1_bound_mhc_r200_fa4_autotune_aiter_hd256_phase181/run.py --check
```
