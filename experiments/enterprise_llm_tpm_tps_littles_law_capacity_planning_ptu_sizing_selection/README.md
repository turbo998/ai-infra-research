# Enterprise LLM TPM vs. TPS Conversion, Little's Law Concurrency Sizing & Cloud PTU/GSU Capacity Planning Benchmark

Deterministic quantitative evaluation model for [`docs/components/enterprise-llm-tpm-tps-littles-law-capacity-planning-ptu-sizing-selection-faq.md`](../../docs/components/enterprise-llm-tpm-tps-littles-law-capacity-planning-ptu-sizing-selection-faq.md) and [`.en.md`](../../docs/components/enterprise-llm-tpm-tps-littles-law-capacity-planning-ptu-sizing-selection-faq.en.md).

## Provenance
- **Label**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)

## Run & Verify
```bash
python experiments/enterprise_llm_tpm_tps_littles_law_capacity_planning_ptu_sizing_selection/run.py
python experiments/enterprise_llm_tpm_tps_littles_law_capacity_planning_ptu_sizing_selection/run.py --check
python -m unittest tests/test_enterprise_llm_tpm_tps_littles_law_capacity_planning_ptu_sizing_selection.py
```
