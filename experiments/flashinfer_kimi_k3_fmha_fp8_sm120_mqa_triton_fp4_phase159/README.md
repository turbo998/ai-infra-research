# FlashInfer Kimi-K3 Dense Routes, One-Stage FP8 Context, SM120 CTA-Parallel MQA & Triton FP4/gfx1250 TDM (Phase 159)

Deterministic offline analytical experiment (`own-offline`, Python 3.12 standard library only) modeling:
1. FlashInfer `cake_kimi_k3_attn_res` dense route retuning on B200/B300/GB300 (`#8120`).
2. FlashInfer `cake_fmha` one-stage multi-wave & short-KV FP8 paged context (`#8101`) + `cake_deepgemm` SM120 CTA-parallel FP8 MQA logits (`#8099`).
3. FlashInfer opt-in non-deterministic MoE grouped GEMM training (`#8088`) & Megatron-LM determinism debugging (`#7262`, `#8149`).
4. Triton AMD `gfx1250` Multi-CTA TDM Scatter (`#12324`), native `fp4`/`e8m0` types (`#12315`), and TileLang Bisheng temp cleanup (`#3340`, `#3461`).
