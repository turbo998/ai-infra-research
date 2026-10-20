# FlashInfer MiniMax-H3 SM120 Epilogue Proxy Fence & Ping-Pong Barrier Close, CAKE NVFP4 SVDQuant 15-Program GEMM Backend, and Rubin MoE `execute_on_streams` API

Deterministic Python 3.12 standard-library simulator (`own-offline`) verifying:
1. `flashinfer-ai/flashinfer@aacd0f1`: MiniMax-H3 SM120 dense attention `fence.proxy.async.shared::cta` placement before `__syncthreads()` (`49 -> 0` cross-warp generic/async proxy hazards across 8 persistent work items) and Group 0 (`warp < 4`) `barrier.sync 1, 256` ping-pong barrier closure (`1 -> 0` open named-barrier rounds at CTA exit).
2. `flashinfer-ai/flashinfer@ec3bcf0`: CAKE NVFP4 SVDQuant 15-program GEMM backend (`out = BF16(alpha * (deq(A) @ deq(B)^T + D @ L1^T) + bias)` across LoRA ranks 32/64/96/128) achieving up to `3.28x` vs previous Cake and `1.293x` vs CUTLASS on B200 (`sm_100a`), and up to `3.58x` vs previous Cake (`1.203x` vs CUTLASS) on GB300 (`sm_103a`).
3. `flashinfer-ai/flashinfer@6445497`: Rubin locality-domain MoE `torch.cuda.execute_on_streams(localized_streams, callback, range(len(localized_streams)))` explicit sequence contract and per-dtype `cp.reduce.async.bulk` copy size (`512 B` BF16 / `1024 B` FP32).

## Run

```bash
python experiments/flashinfer_minimax_h3_sm120_epilogue_nvfp4_svdquant_moe/run.py --check
```
