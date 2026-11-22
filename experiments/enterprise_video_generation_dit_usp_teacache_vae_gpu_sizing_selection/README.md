# Enterprise Video Generation (`14B DiT`: `Wan2.1 / HunyuanVideo` Class) Spatiotemporal Token Sizing, `xDiT (CFG × Ulysses)` & `FP8 + TeaCache` Selection (`#472`)

Reproducible analytical model comparing Single-GPU BF16 sequential denoising, `TP=8` DiT parallelism, `xDiT (CFG=2 × Ulysses=4)` sequence parallelism, and `xDiT + FP8 + TeaCache + Parallel Tiled 3D VAE` on `720p 81-frame` (`32,400` latent tokens) video generation.

- **Provenance**: `synthetic + anonymized_reported (independently_reproduced=false)`
- **Run**: `python experiments/enterprise_video_generation_dit_usp_teacache_vae_gpu_sizing_selection/run.py --check`
