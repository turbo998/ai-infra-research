# Video Generation DiT Infrastructure Selection Experiment (`Wan2.1 / HunyuanVideo` with `xDiT USP + CFG Parallel + FP8 SageAttention + TeaCache`)

Reproducible performance and TCO model comparing 4 parallel inference architectures for `14B`-parameter 3D Video Diffusion Transformers (`72,000` spatio-temporal tokens, `50` denoising steps).

## Run
```bash
python experiments/video_generation_dit_wan21_hunyuan_usp_ulysses_ring_fp8_tea_cache_selection/run.py
python experiments/video_generation_dit_wan21_hunyuan_usp_ulysses_ring_fp8_tea_cache_selection/run.py --check
```
