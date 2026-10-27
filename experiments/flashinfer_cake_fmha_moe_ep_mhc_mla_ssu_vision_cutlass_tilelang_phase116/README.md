# FlashInfer Cake FMHA Split-KV FP8 / MegaMoE Autotune v2 / SM107a Mega-mHC / CuTe-DSL MLA Split Guard / SSU Ring / MoonViT-3D & TileLang Lower-Trace Model (Phase 116)

Reproducible CPU/offline analytical model (`evidence_mode: own-offline`) for Phase 116 Track A (`priority: 125`):
- FlashInfer Cake FMHA split-KV FP8 paged context static first tile & direct device scales (`a8803f2c`)
- FlashInfer MegaMoE Blackwell standard & GenPhase `autotune_v2` (`d65f7323`) and `sm_107a` BF16 Mega-mHC two-warp pair fold (`b4b38b0b`)
- FlashInfer CuTe-DSL MLA decode 1024-token/split streaming floor (`9a2aea3f`), Cake Mamba2 SSU FP32 L2 read-ahead / ring (`593b468b`), and Kimi-K3 MoonViT-3D vision tower (`87dbb060`)
- CUTLASS SM100f distributed GEMM & FMHA fixes (`e64a9136`, `2802e228`) and TileLang autotuner `ProfileArgs` forwarding & host/device lower-trace separation (`90f8908b`, `ab05a561`)
