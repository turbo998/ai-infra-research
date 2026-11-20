# Enterprise Supply Chain Demand Forecasting: Chronos/TimesFM + LightGBM, LLM Covariate Featurizer & Conformal Safety-Stock Benchmark

Deterministic quantitative evaluation model for [`docs/components/enterprise-supply-chain-demand-forecasting-time-series-foundation-erp-selection-faq.md`](../../docs/components/enterprise-supply-chain-demand-forecasting-time-series-foundation-erp-selection-faq.md) and [`.en.md`](../../docs/components/enterprise-supply-chain-demand-forecasting-time-series-foundation-erp-selection-faq.en.md).

## Provenance
- **Label**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)

## Run & Verify
```bash
python experiments/enterprise_supply_chain_demand_forecasting_time_series_foundation_erp_selection/run.py
python experiments/enterprise_supply_chain_demand_forecasting_time_series_foundation_erp_selection/run.py --check
python -m unittest tests/test_enterprise_supply_chain_demand_forecasting_time_series_foundation_erp_selection.py
```
