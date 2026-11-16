# PyTorch AMP `GradScaler` on `BF16` (`found_inf` Sync Stall) vs. `FP8` Blockwise (`1x128` / `128x128`) JIT Scaling Evaluator

Deterministic evaluator for Round 348 (`pytorch-amp-gradscaler-bf16-found-inf-sync-stall-fp8-scaling-selection-faq`).

- **Provenance**: `synthetic + anonymized_reported (independently_reproduced=false, physical_gpu_cluster_measurement=not_measured)`
- **Run**:
  ```bash
  python experiments/pytorch_amp_gradscaler_bf16_found_inf_sync_stall_fp8_scaling/run.py
  ```
