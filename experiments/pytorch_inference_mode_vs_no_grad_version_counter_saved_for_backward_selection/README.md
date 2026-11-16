# PyTorch `torch.inference_mode()` vs. `torch.no_grad()` Version-Counter Dispatch Overhead & `SavedVariable` Backward Crash Evaluator

> **Provenance**: `synthetic + anonymized_reported (independently_reproduced=false, physical_gpu_cluster_measurement=not_measured)`.

Reproduces the sealed metrics for `docs/components/pytorch-inference-mode-vs-no-grad-version-counter-saved-for-backward-selection-faq.md`:
- **Tier 1 (`with torch.no_grad():` across `960` op dispatches)**: `4.85 ms` CPU dispatch time because `no_grad()` still allocates atomic `VariableVersion` counters and `DifferentiableViewMeta` on all `960` tensors.
- **Tier 2 (Raw `@torch.inference_mode()` output passed directly to a trainable `nn.Linear`)**: `2.90 ms` dispatch, **`100.0%` crash** (`RuntimeError: Inference tensors cannot be saved for backward`).
- **Tier 3 (Exiting `inference_mode()` without calling `.clone()` outside the context)**: `100.0%` crash because `is_inference() == True` is a permanent tensor property set at creation time.
- **Tier 4 (`with torch.inference_mode():` inside the frozen tower/rollout + single exit-boundary `out = out.clone()` outside the context)**: `2.92 ms` total dispatch (`1.66x` faster than `no_grad()`), `1` version counter instead of `960`, and `0.0%` crash rate.
