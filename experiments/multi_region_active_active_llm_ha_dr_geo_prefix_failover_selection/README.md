# Multi-Region Active-Active LLM HA/DR & Global Traffic Routing: Geo-Sticky Prefix Affinity & Cascading-Overload Prevention Benchmark

Deterministic quantitative evaluation model for [`docs/components/multi-region-active-active-llm-ha-dr-geo-prefix-failover-selection-faq.md`](../../docs/components/multi-region-active-active-llm-ha-dr-geo-prefix-failover-selection-faq.md) and [`.en.md`](../../docs/components/multi-region-active-active-llm-ha-dr-geo-prefix-failover-selection-faq.en.md).

## Provenance
- **Label**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)

## Run & Verify
```bash
python experiments/multi_region_active_active_llm_ha_dr_geo_prefix_failover_selection/run.py
python experiments/multi_region_active_active_llm_ha_dr_geo_prefix_failover_selection/run.py --check
python -m unittest tests/test_multi_region_active_active_llm_ha_dr_geo_prefix_failover_selection.py
```
