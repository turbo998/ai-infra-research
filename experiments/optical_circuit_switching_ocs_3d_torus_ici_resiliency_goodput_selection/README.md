# 10K-Chip Optical Circuit Switching (Palomar 3D MEMS OCS), 3D Torus Defragmentation & ICI Resiliency Benchmark

Deterministic quantitative evaluation model for [`docs/components/optical-circuit-switching-ocs-3d-torus-ici-resiliency-goodput-selection-faq.md`](../../docs/components/optical-circuit-switching-ocs-3d-torus-ici-resiliency-goodput-selection-faq.md) and [`.en.md`](../../docs/components/optical-circuit-switching-ocs-3d-torus-ici-resiliency-goodput-selection-faq.en.md).

## Provenance
- **Label**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)

## Run & Verify
```bash
python experiments/optical_circuit_switching_ocs_3d_torus_ici_resiliency_goodput_selection/run.py
python experiments/optical_circuit_switching_ocs_3d_torus_ici_resiliency_goodput_selection/run.py --check
python -m unittest tests/test_optical_circuit_switching_ocs_3d_torus_ici_resiliency_goodput_selection.py
```
