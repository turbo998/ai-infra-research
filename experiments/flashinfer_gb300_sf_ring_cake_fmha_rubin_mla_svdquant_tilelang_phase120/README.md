# Phase 120 Track A Experiment: FlashInfer GB300 SF-Ring FP4, CAKE FMHA Paged Decode, Rubin MLA, SVDQuant Tail, B12X MoE, cuDNN 1.32 QSA & TileLang ROCm 10

Reproducible Python standard-library model (`run.py`) and sealed output (`summary.json`) covering 10 upstream commits from `flashinfer-ai/flashinfer` and `tile-ai/tilelang`:

- `flashinfer-ai/flashinfer@4ca4e6e`: GB300 (`sm_103a`) `cake_mm_fp4` separate 5-stage scale-factor ring (`1.1061x` peak GEMM speedup, `1.0836x` `quant+GEMM` speedup).
- `flashinfer-ai/flashinfer@5e0e99b`: `cake_fmha` 27 paged decode attention programs (`1.750x` B200 BF16 speedup, `353.17x` device-scale launch speedup).
- `flashinfer-ai/flashinfer@f6587bc`: `cake_rubin_mla_nvfp4_decode` (`sm_107a`) widened PDL trigger (`50 -> 54` items) and `probe_after_loads` / `column` merge (`1.067x`).
- `flashinfer-ai/flashinfer@03ebbc6`: `cake_nvfp4_svdquant` one-wave tail build (`8*192 + 68*128 = 10240` cols across 76 CTA pairs, `1.0644x` GB200 speedup).
- `flashinfer-ai/flashinfer@d949946` & `7de971a`: `cake_minimax_h3_varlen_attention` (`2.17x–2.34x` tiny BF16) and `cake_warp_decode` count-rank route-pack (`1.0635x`).
- `flashinfer-ai/flashinfer@959a113` & `c2ba71f`: `b12x` MoE power-of-two workspace bucketing (`39.67x` fewer JIT compiles) and cuDNN 1.32 QSA forward APIs.
- `tile-ai/tilelang@040b305` & `015caa2`: ROCm `current_device()` target lookup and 4-tier ROCm 10 pip-SDK architecture detection.

## Usage

```bash
python experiments/flashinfer_gb300_sf_ring_cake_fmha_rubin_mla_svdquant_tilelang_phase120/run.py --check
```
