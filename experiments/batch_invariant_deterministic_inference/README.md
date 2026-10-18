# Batch-Invariant Deterministic Inference Simulator (`experiments/batch_invariant_deterministic_inference`)

Reproduces floating-point non-associativity drift (`8–12 ULPs` across 80 layers and `96.55%–99.36%` 1,000-step CoT trajectory divergence) when dynamic batch sizes (`BS=1, 8, 33, 64, 128`) change `num_splits_heuristic` in FlashAttention (`csrc/flash_attn/flash_api.cpp`) and Split-K in GEMM, and verifies `0 ULP` (`0.00%` divergence) under Batch-Invariant Fixed-Tree Kernels (`+4.21%–8.57%` TPOT overhead).

## Run & Verify

```powershell
python experiments\batch_invariant_deterministic_inference\run.py --check
```
