# FlashInfer KDA 2-CTA CP State Walks, BF16×FP4 TMEM/Scale Opt, MXFP8 Quantize & TileLang SM120 GEMM (Phase 88)

Deterministic Python 3.12 standard-library verifier covering:
- **FlashInfer `#7415` (`182159eb7a`)**: `cake_warp_decode` fused route-pack FC1 + K512 MMA-U2 FC2 on 100 `sm_103a` rows, fixed top-10 vector finalize on 20 rows, and linear-grid K256 FC2 on 24 `h3072_e256_i384_k8` rows.
- **FlashInfer `#7436` (`4f14b25dc6`)**: `cake_kda` context-parallel two-CTA state walk records (64 columns per CTA, 2 CTAs/SM on GB300, up to `1.0754x` forward / `1.0512x` fwd+replay+bwd across 144 table rows).
- **FlashInfer `#7419` (`328e862b34`)**: `cake_blackwell_bf16_fp4` early pair TMEM deallocation (`1.099x` GB300) and pre-loop block-scale conversion (`1.205x` B200).
- **FlashInfer `#7431` (`bdaa2401ad`)**: `cake_grouped_mxfp8_quantize` per-K row-tiled programs (`K=4096` 256-thread 2-row CTA, `K=2048` 128-thread CTA) and integer UE8M0 scale extraction (`1.3632x` GB200).
- **FlashInfer `#7407` (`fdd121f46b`), `#7417` (`a8d23b20bf`), `#6885` (`0af9594727`), `#7434` (`9e656abf02`)**: Mamba2 `selective_state_update` P-split/TMA-ring, `cake_rmsnorm_train` slab reduction (`1.18x` on `sm_107a`), `cake_all_gather_matmul` CTA-pair producer tail barrier fix, and `MiniMaxH3Nvfp4PreAttention` (`1.1198x` GB300).
- **TileLang `#3286` (`32f56122e2`) & `#3649` (`89b5b5b9ad`)**: SM120 block-scaled register-A GEMM K-atom unroll + compact scale selector package (`6.0x` fewer shared scale loads) and caller launch-stream output allocation.

## Run
```bash
python experiments/flashinfer_kda_2cta_bf16_fp4_mxfp8_ssu_phase88/run.py
python experiments/flashinfer_kda_2cta_bf16_fp4_mxfp8_ssu_phase88/run.py --check
```
