# Enterprise Semiconductor & Electronics Fab Yield RCA, Wafer Map Clustering, AOI & Virtual Metrology Agent (`R437`)

Deterministic evaluation suite comparing four semiconductor/electronics manufacturing yield root-cause analysis (`RCA`), Automated Optical Inspection (`AOI`), Virtual Metrology (`VM`), and Run-to-Run (`R2R`) control architectures across a `45,000 wafers/month` fab (`600` process steps, `1,200` tool chambers, `$8,500` value/wafer):

1. **Tier 1**: Manual Yield Engineer Excel + Univariate SPC (`36.0h` excursion RCA, `18.5%` AOI false overkill, `7.8%` unmeasured lot scrap escape, `52.0%` chamber hit rate, `$940,000/mo` scrap & metrology cost)
2. **Tier 2**: Prompt-Only General LLM FDC Alarm Log Summarizer (`14.0h` RCA, `21.0%` AOI overkill, `8.4%` scrap escape, `44.5%` chamber hit rate, `$1,050,000/mo` cost)
3. **Tier 3**: Standalone Wafer-Map CNN Without Chamber Genealogy (`6.5h` RCA, `6.4%` AOI overkill, `3.9%` scrap escape, `74.0%` chamber hit rate, `$410,000/mo` cost)
4. **Tier 4**: Polar Wafer-Map `DBSCAN/ViT` + 600-Step Chamber Genealogy Graph + Reliance-Index (`RI >= 0.85`) Virtual Metrology + `SECS/GEM EWMA R2R` Clamp (`0.35h` / 21-min RCA, `0.6%` AOI overkill, `0.0%` scrap escape, `98.4%` chamber hit rate, `$115,000/mo` TCO)

## Usage

```bash
python experiments/enterprise_semiconductor_fab_wafer_yield_aoi_virtual_metrology_agent_selection/run.py
python experiments/enterprise_semiconductor_fab_wafer_yield_aoi_virtual_metrology_agent_selection/run.py --check
python -m unittest tests/test_enterprise_semiconductor_fab_wafer_yield_aoi_virtual_metrology_agent_selection.py
```
