# Phase 110 Track A: FlashInfer Fused Norm Combine, MiniMax-H3 SM120 Pre-Attn GEMM, NVFP4 L2 Prefetch, Kimi-K3 TP12 Tail & MLA, MoE AR Union, CUTLASS & TileLang

Deterministic stdlib model of Phase 110 (`priority: 119`) Track A kernel and compiler changes:
- `flashinfer` (`3da22ba71f`, `5b2aacd211`, `e158799310`, `531e0bc2ba`, `45411ad7d7`, `acb2bef7ec`)
- `cutlass` (`976d0521f0`, `3c063071bd`)
- `tilelang` (`cb6455d680`, `9253431f2a`)

Run `python experiments/flashinfer_norm_combine_minimax_fp4_kimi_mla_phase110/run.py --check` to verify `summary.json`.
