# Enterprise Energy Grid & Virtual Power Plant (`100,000 Wind/Solar/BESS Assets`): `Legacy NWP` vs. `Direct LLM Bidding` vs. `GraphCast/Aurora AI Weather + TimesFM + Stochastic MILP Solver` (`#548`)

- **Chinese FAQ**: [`../../docs/components/enterprise-energy-grid-weather-tsfm-stochastic-milp-bidding-sizing-faq.md`](../../docs/components/enterprise-energy-grid-weather-tsfm-stochastic-milp-bidding-sizing-faq.md)
- **English FAQ**: [`../../docs/components/enterprise-energy-grid-weather-tsfm-stochastic-milp-bidding-sizing-faq.en.md`](../../docs/components/enterprise-energy-grid-weather-tsfm-stochastic-milp-bidding-sizing-faq.en.md)
- **Interactive H5 Lab**: [`../../docs/h5/energy-grid-weather-tsfm-stochastic-milp-bidding-lab.html`](../../docs/h5/energy-grid-weather-tsfm-stochastic-milp-bidding-lab.html)
- **Interactive Sizer**: [`../../docs/reports/enterprise-energy-grid-weather-tsfm-stochastic-milp-bidding-sizing-sizer.html`](../../docs/reports/enterprise-energy-grid-weather-tsfm-stochastic-milp-bidding-sizing-sizer.html)
- **7-Card Report**: [`../../docs/reports/enterprise-energy-grid-weather-tsfm-stochastic-milp-bidding-sizing-cards.html`](../../docs/reports/enterprise-energy-grid-weather-tsfm-stochastic-milp-bidding-sizing-cards.html)

## Reproduce
```bash
python experiments/enterprise_energy_grid_weather_tsfm_stochastic_milp_bidding_sizing/run.py
python -m pytest tests/test_enterprise_energy_grid_weather_tsfm_stochastic_milp_bidding_sizing.py -q
```
