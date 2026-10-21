# FlashInfer Cake KDA TF32, FP8 GEMM, StepFun MoE, MiniMax H3 & cuDNN Paged Prefill HND Verification

Deterministic Python 3.12 standard-library verification of:
- `flashinfer-ai/flashinfer@9f7a290987b69277a9031b1c6091810d83cfed3b` (`#6876`): Cake KDA TF32 24->9 kernel folding (`97,009 -> 41,231` lines, `-57.50%`).
- `flashinfer-ai/flashinfer@272ed2a6cf2fe66356b31b91635c26220b819fc8` (`#6862`): Cake FP8 GEMM 4-family consolidation (`60` programs, 56-row routing table, batch loader `6,381 -> 1,548` lines).
- `flashinfer-ai/flashinfer@e1b052de722e193324559ba2639b26992c7b9f50` (`#6871`): Cake StepFun SwiGLU fused-MoE (`111` programs x `2` archs = `222` manifest entries, `47` shared bindings).
- `flashinfer-ai/flashinfer@5ea501832bf7e37a7acde903a32329d580c0e9b4` (`#6869`): MiniMax H3 QKV quantize+pack (`8` variants, `1.000x` B200/GB300 parity).
- `flashinfer-ai/flashinfer@013f02e75ddf8430ad010b965b2d5047c370f576` (`#6846`): cuDNN paged prefill HND layout contract (`[pages, heads=8, page_size=16, dim]`).

## Run
```bash
python experiments/flashinfer-cake-kda-tf32-fp8-gemm-stepfun-moe-minimax-h3-cudnn-hnd/run.py --check
```
