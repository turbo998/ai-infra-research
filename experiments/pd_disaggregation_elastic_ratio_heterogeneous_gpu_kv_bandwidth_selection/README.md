# Experiment: PD Disaggregation Dynamic P:D Ratio Sizing, Heterogeneous GPU Pairing & Cross-Node KV Bandwidth (`#396`)

Reproducible analytical sizing model comparing four Prefill-Decode (PD) disaggregation and KV transfer architectures across diurnal RAG (`16K In / 400 Out`) and Deep Reasoning (`2K In / 4K Out`) workloads.

- **Provenance**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)
- **Run**: `python experiments/pd_disaggregation_elastic_ratio_heterogeneous_gpu_kv_bandwidth_selection/run.py`
- **Verify**: `python experiments/pd_disaggregation_elastic_ratio_heterogeneous_gpu_kv_bandwidth_selection/run.py --check`
