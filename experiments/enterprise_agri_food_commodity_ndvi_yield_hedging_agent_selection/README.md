# Enterprise Agricultural & Food Commodity SAR/NDVI Yield Forecasting & CVaR Hedging Agent (`R435`)

Deterministic evaluation suite comparing four agricultural commodity procurement, satellite crop-yield forecasting, and silo spoilage-prevention architectures across a `$600M/yr` global food & beverage procurement operation (`18,500` crop polygons, `80` grain/coffee silos, `8-week` pre-harvest lead time):

1. **Tier 1**: Lagged Official Monthly Reports + 5-Year Moving Average Excel (`16.8%` yield MAPE, `38.0` cloud-blind days, `24.5%` tail price spike loss, `9.4%` silo spoilage escape, `$620,000/mo` loss)
2. **Tier 2**: Prompt-Only General LLM Commodity News Sentiment Trader (`21.4%` yield MAPE, `30.0` blind days, `29.0%` tail price loss, `11.2%` spoilage escape, `$710,000/mo` loss)
3. **Tier 3**: Optical-Only Satellite NDVI Regression (`9.6%` yield MAPE, `24.0` monsoon cloud-blind days, `12.8%` tail loss, `4.8%` spoilage escape, `$295,000/mo` loss)
4. **Tier 4**: `Sentinel-1 C-Band SAR` + `Sentinel-2 NDVI` Fusion + `ECMWF` Ensemble VPD Agro-Physics + Two-Stage Stochastic `CVaR` Basis Hedging + Silo `Henderson EMC` Gate (`2.9%` yield MAPE, `0.0` cloud-blind days, `1.8%` tail price loss, `0.0%` silo mycotoxin escape, `$58,000/mo` TCO)

## Usage

```bash
python experiments/enterprise_agri_food_commodity_ndvi_yield_hedging_agent_selection/run.py
python experiments/enterprise_agri_food_commodity_ndvi_yield_hedging_agent_selection/run.py --check
python -m unittest tests/test_enterprise_agri_food_commodity_ndvi_yield_hedging_agent_selection.py
```
