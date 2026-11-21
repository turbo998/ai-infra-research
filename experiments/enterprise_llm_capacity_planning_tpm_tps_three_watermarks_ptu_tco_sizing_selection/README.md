# Experiment: Enterprise LLM Capacity Planning & Three-Watermark GPU Sizing Master Guide (`#399`)

Reproducible analytical model unifying business DAU/RPS, Input/Output TPM & TPS, Cloud PTU weighted conversion, Little's Law concurrency, Three-Watermark GPU sizing (`max[N_HBM, N_BW, N_Compute]`), and P80/P20 hybrid spillover TCO.

- **Provenance**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)
- **Run**: `python experiments/enterprise_llm_capacity_planning_tpm_tps_three_watermarks_ptu_tco_sizing_selection/run.py`
- **Verify**: `python experiments/enterprise_llm_capacity_planning_tpm_tps_three_watermarks_ptu_tco_sizing_selection/run.py --check`
