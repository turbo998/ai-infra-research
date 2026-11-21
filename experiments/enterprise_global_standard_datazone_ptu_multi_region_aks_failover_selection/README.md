# Experiment: Enterprise Multi-Region HA/DR & Cloud Deployment Tier Selection — Global Standard vs. DataZone vs. PTU vs. Multi-Region AKS (`#405`)

Reproducible analytical model evaluating `Regional Standard`, `DataZone Standard (EU/US)`, `Global Standard`, `Provisioned Throughput (PTU)`, and `Multi-Region Active-Active AKS (KAITO / vLLM)` under 3x burst spikes and regional failover drills.

- **Provenance**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)
- **Run**: `python experiments/enterprise_global_standard_datazone_ptu_multi_region_aks_failover_selection/run.py`
- **Verify**: `python experiments/enterprise_global_standard_datazone_ptu_multi_region_aks_failover_selection/run.py --check`
