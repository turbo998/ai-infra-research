# Enterprise Retail Chain Store Site Selection, Geo-Spatial Huff Gravity & Commercial Lease Review Agent (`R431`)

Deterministic evaluation suite comparing four retail expansion and commercial real-estate underwriting architectures across a 1,500-store chain evaluating 350 candidate locations per month (240 new openings/year at `$420K` CapEx per store):

1. **Tier 1**: Broker Intuition + Static 1km Euclidean Ring Excel (`34.5%` Year-1 Revenue MAPE, `28.0%` cannibalization error, `18.5%` lease trap escape, `$410,000/mo` underwriting & bad-site loss)
2. **Tier 2**: Prompt-Only General LLM Neighborhood Recommender (`41.2%` Year-1 Revenue MAPE, `35.0%` cannibalization error, `11.4%` lease trap escape, `$495,000/mo` loss)
3. **Tier 3**: Generic Tabular Census Regression (`19.8%` Year-1 Revenue MAPE, `17.5%` cannibalization error, `8.2%` lease trap escape, `$215,000/mo` loss)
4. **Tier 4**: Uber `H3 Res-9` Isochrone Grid + Calibrated `Huff Gravity` Cannibalization Solver + Street-View Storefront `VLM` + Commercial Lease Clause `AST` Gate (`6.4%` Year-1 Revenue MAPE, `2.8%` cannibalization error, `0.0%` lease trap escape, `0.25 days` underwriting cycle, `$38,000/mo` TCO)

## Usage

```bash
python experiments/enterprise_retail_site_selection_geospatial_huff_lease_agent_selection/run.py
python experiments/enterprise_retail_site_selection_geospatial_huff_lease_agent_selection/run.py --check
python -m unittest tests/test_enterprise_retail_site_selection_geospatial_huff_lease_agent_selection.py
```
