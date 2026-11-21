# JAX/XLA Shardy (MLIR) vs. GSPMD Partitioner, Hierarchical Device Mesh & ICI-vs-DCN Parallelism Benchmark

Deterministic quantitative evaluation model for [`docs/components/jax-xla-shardy-gspmd-device-mesh-ici-dcn-parallelism-selection-faq.md`](../../docs/components/jax-xla-shardy-gspmd-device-mesh-ici-dcn-parallelism-selection-faq.md) and [`.en.md`](../../docs/components/jax-xla-shardy-gspmd-device-mesh-ici-dcn-parallelism-selection-faq.en.md).

## Provenance
- **Label**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)

## Run & Verify
```bash
python experiments/jax_xla_shardy_gspmd_device_mesh_ici_dcn_parallelism_selection/run.py
python experiments/jax_xla_shardy_gspmd_device_mesh_ici_dcn_parallelism_selection/run.py --check
python -m unittest tests/test_jax_xla_shardy_gspmd_device_mesh_ici_dcn_parallelism_selection.py
```
