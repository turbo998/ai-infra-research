# FlashInfer Frost DeepSeek-V4.1 Mixed MXFP8/FP4 Decode & Green Context Simulator (`flashinfer_ds41_mixed_decode_sim`)

Deterministic Python 3.12 standard-library model of FlashInfer's SM100 Frost DeepSeek-V4.1 mixed-cache decode (`flashinfer/deepseek_v41.py`, `flashinfer/experimental/deepseek_v41/decode.py`, `flashinfer/experimental/deepseek_v41/cache.py`, and `flashinfer/green_ctx.py`, commit `8dc1163fbfbec7098a2418276fd04bc1fa1aeaa5`).

## What It Models

1. **Non-Interleaved Page Pool Layout (`swa_mxfp8` 528B vs `main_kv_fp4` 288B)**:
   - Verifies exact byte offsets `base = (slot // 64) * 64 * (WIDTH + SF)`, `data_base = base + (slot % 64) * WIDTH`, and `scale_base = base + 64 * WIDTH + (slot % 64) * SF`.
   - Demonstrates why `cache.py` uses `tl.div_rn(ga, scale)` instead of reciprocal multiplication `ga * (1.0 / scale)` before `cvt.rn.satfinite.e2m1x2.f32`.
2. **Adaptive Split-KV / Warp-Specialized (`use_ws`) / Persistent-CTA Planner**:
   - Models the transition from small-batch Split-KV (`split = min(desired, (128 + ck) // 64)`) to large-batch `dequant_warps=16` Warp Specialization (`batch >= 128, ck == 512`) and wave-balanced `persistent_ctas = ceil(batch / ceil(batch / sm_count))`.
3. **Sink-Exclusive Public LSE & Final Sink Normalization**:
   - Verifies that `lse` (`FP32[B, 64, 1]`) remains natural-log and sink-exclusive across Split-K log2 merges while the attention sink (`FP32[64]`) is applied once during final probability normalization.
4. **CUDA Green Context SM Partitioning (`green_ctx.py`)**:
   - Models `get_sm_count_constraint(major >= 9) -> (8, 8)` alignment and iterative SM splitting.

## Commands

```bash
python experiments/flashinfer_ds41_mixed_decode_sim/run.py
python experiments/flashinfer_ds41_mixed_decode_sim/run.py --check
```
