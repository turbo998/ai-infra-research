# PyTorch `AdamW` Optimizer Execution Mode (`foreach=True` `17.50 GB` Temp Buffer OOM vs. `fused=True, capturable=True` & Device-Tensor `lr`) Evaluator

> **Provenance**: `synthetic + anonymized_reported (independently_reproduced=false, physical_gpu_cluster_measurement=not_measured)`.

Reproduces the sealed metrics for `docs/components/pytorch-adamw-foreach-temp-buffer-oom-fused-capturable-selection-faq.md`:
- **Tier 1 (Single-Tensor `for`-loop, `foreach=False, fused=False`)**: `4,350` kernel launches (`725` parameter tensors × `6` ops), `28.40 ms` step latency.
- **Tier 2 (Multi-Tensor `foreach=True, fused=False`)**: `12` multi-tensor kernels, `8.60 ms` step latency, **`17.50 GB` temporary buffer allocation spike** from out-of-place `torch._foreach_sqrt(exp_avg_sq)`.
- **Tier 3 (`fused=True, capturable=False` + Python float `lr`)**: `0.00 GB` temp buffer, `4.10 ms` step latency, `64` `torch.compile` recompilations per `1,000` steps on cosine LR updates and incompatible with CUDA Graph capture.
- **Tier 4 (`fused=True, capturable=True` + 0-D CUDA Tensor `lr`)**: `0.00 GB` temp buffer (`-17.50 GB` peak HBM saved vs. `foreach=True`), `3.80 ms` step latency (`7.5x` faster than Tier 1, `2.26x` faster than Tier 2), `0` recompilations.
