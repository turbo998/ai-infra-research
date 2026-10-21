# FlashInfer Cake GLM-5.3 Four-Stream mHC, Blackwell MSA K-Major Prefill & TRTLLM-Gen Multi-Wave KV Split / BMM Ordering

Deterministic verification script for:
- `flashinfer-ai/flashinfer@130dd2429a` (PR `#6897`): Cake four-stream mHC training kernels (`flashinfer.cake_mhc_train`) on SM100/SM103/SM107.
- `flashinfer-ai/flashinfer@2ff6d8a5f2` (PR `#6873`) & `3b65505f40` (PR `#6904`): Cake Blackwell MSA persistent K-major prefill (`3.39x` GB300 geomean) and Round-2 Kimi-K3 AttnRes.
- `flashinfer-ai/flashinfer@c7d1eacfdc` (PR `#6875`), `a3f2525e39` (PR `#6331`), & `c1bcf70660` (PR `#6845`): Cake FP4 AllReduce in-flight row-scale read & TP4 SMEM pass-2, TRTLLM-Gen opt-in 4-wave KV split decode, and manifest-independent BMM default config comparator.

## Run
```bash
python experiments/flashinfer_cake_mhc_msa_kmajor_multiwave_bmm/run.py --check
```
