# Enterprise GPU Cluster Power Capping, Phase-Aware DVFS & Rack Density Selection Benchmark

Deterministic quantitative evaluation model for [`docs/components/enterprise-gpu-cluster-power-capping-dvfs-thermal-slo-scheduling-selection-faq.md`](../../docs/components/enterprise-gpu-cluster-power-capping-dvfs-thermal-slo-scheduling-selection-faq.md) and [`.en.md`](../../docs/components/enterprise-gpu-cluster-power-capping-dvfs-thermal-slo-scheduling-selection-faq.en.md).

## Provenance
- **Label**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)

## Run & Verify
```bash
python experiments/enterprise_gpu_cluster_power_capping_dvfs_thermal_slo_scheduling_selection/run.py
python experiments/enterprise_gpu_cluster_power_capping_dvfs_thermal_slo_scheduling_selection/run.py --check
python -m unittest tests/test_enterprise_gpu_cluster_power_capping_dvfs_thermal_slo_scheduling_selection.py
```
