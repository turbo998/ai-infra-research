# Milestone `#530`: Enterprise Video DiT Serving (`xDiT USP Sequence Parallelism + TeaCache + SageAttention2 FP8`) Bible

Reproducible analytical model comparing **Single-GPU BF16 50-Step Video DiT (`340.0s` per 720p clip)**, **8-GPU `xDiT` Unified Sequence Parallelism (`Ulysses=4 × Ring=2`, `46.5s`)**, and **`USP=8` + `TeaCache` Timestep Residual Caching + `SageAttention2 FP8` (`11.2s` per clip [`30.36x` faster], `321.4 clips/hr/node` [`3.79x` throughput], `$0.077/clip` [`-73.7%` cost] 🏆)**.

## Run

```bash
python experiments/enterprise_milestone_530_video_dit_serving_usp_teacache_fp8_sizing_bible/run.py
```

- **Provenance**: `synthetic + anonymized_reported (independently_reproduced=false)`
- **Bilingual Guides**:
  - [Chinese FAQ](../../docs/components/enterprise-milestone-530-video-dit-serving-usp-teacache-fp8-sizing-bible-faq.md)
  - [English FAQ](../../docs/components/enterprise-milestone-530-video-dit-serving-usp-teacache-fp8-sizing-bible-faq.en.md)
- **Interactive H5 Lab**: [docs/h5/milestone-530-video-dit-serving-usp-teacache-fp8-lab.html](../../docs/h5/milestone-530-video-dit-serving-usp-teacache-fp8-lab.html)
