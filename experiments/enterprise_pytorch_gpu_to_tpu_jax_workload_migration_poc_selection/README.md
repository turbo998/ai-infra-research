# Enterprise PyTorch/GPU to TPU/JAX Workload Migration: 3-Lane Triage, XLA De-Trapping & 14-Day PoC Benchmark

Deterministic quantitative evaluation model for [`docs/components/enterprise-pytorch-gpu-to-tpu-jax-workload-migration-poc-selection-faq.md`](../../docs/components/enterprise-pytorch-gpu-to-tpu-jax-workload-migration-poc-selection-faq.md) and [`.en.md`](../../docs/components/enterprise-pytorch-gpu-to-tpu-jax-workload-migration-poc-selection-faq.en.md).

## Provenance
- **Label**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)

## Run & Verify
```bash
python experiments/enterprise_pytorch_gpu_to_tpu_jax_workload_migration_poc_selection/run.py
python experiments/enterprise_pytorch_gpu_to_tpu_jax_workload_migration_poc_selection/run.py --check
python -m unittest tests/test_enterprise_pytorch_gpu_to_tpu_jax_workload_migration_poc_selection.py
```
