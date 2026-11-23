# Enterprise Retail Demand Forecasting (`100M SKU-Store Pairs`): `Pure LLM CSV Prompting` vs. `Per-SKU ARIMA/Prophet` vs. `Offline 8B LLM Promo Extractor + Chronos-Bolt/TimesFM Cold-Start + GPU LightGBM` (`#543`)

- **Chinese FAQ**: [`../../docs/components/enterprise-retail-demand-forecasting-tsfm-gbdt-llm-promo-sizing-faq.md`](../../docs/components/enterprise-retail-demand-forecasting-tsfm-gbdt-llm-promo-sizing-faq.md)
- **English FAQ**: [`../../docs/components/enterprise-retail-demand-forecasting-tsfm-gbdt-llm-promo-sizing-faq.en.md`](../../docs/components/enterprise-retail-demand-forecasting-tsfm-gbdt-llm-promo-sizing-faq.en.md)
- **Interactive H5 Lab**: [`../../docs/h5/retail-demand-forecasting-tsfm-gbdt-llm-promo-lab.html`](../../docs/h5/retail-demand-forecasting-tsfm-gbdt-llm-promo-lab.html)
- **Interactive Sizer**: [`../../docs/reports/enterprise-retail-demand-forecasting-tsfm-gbdt-llm-promo-sizing-sizer.html`](../../docs/reports/enterprise-retail-demand-forecasting-tsfm-gbdt-llm-promo-sizing-sizer.html)
- **7-Card Report**: [`../../docs/reports/enterprise-retail-demand-forecasting-tsfm-gbdt-llm-promo-sizing-cards.html`](../../docs/reports/enterprise-retail-demand-forecasting-tsfm-gbdt-llm-promo-sizing-cards.html)

## Reproduce
```bash
python experiments/enterprise_retail_demand_forecasting_tsfm_gbdt_llm_promo_sizing/run.py
python -m pytest tests/test_enterprise_retail_demand_forecasting_tsfm_gbdt_llm_promo_sizing.py -q
```
