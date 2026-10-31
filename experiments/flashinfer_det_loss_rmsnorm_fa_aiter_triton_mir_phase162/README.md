# FlashInfer CAKE Det Loss & B200/GB300/R200 RMSNorm Retune, FlashAttention ROCm System AITER & Triton AMD MIR CodeGen (Phase 162)

Deterministic standard-library analytical model (`own-offline`) for Phase 162 (`priority: 171`) Track 1:
- **FlashInfer**: `785e84e4dc01` (#8126 per-thread hidden-count state + layout-keyed logprob bwd), `aa9932c869bb` (#8059 chunked LM-head det loss names), `95c1a0953651` (#8094 B200/GB300/R200 Muse Spark & MoE finalize RMSNorm retuning), `cf6af475fddf` (#8054 KDA det names).
- **FlashAttention / CUTLASS / Triton**: `9c6d9297e5d4`, `edb5c76ee329`, `1bda8f9290cd`, `dcf215af68a2`, `9245be15dbae`, `af9a9149afb4`, `ff868dc6596c`, `ad50bab56a7d`.
