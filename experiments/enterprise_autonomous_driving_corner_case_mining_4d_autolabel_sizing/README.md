# Enterprise Autonomous Driving Data Closed-Loop (`10,000 Vehicles`): `Raw Upload + Manual 3D Labeling` vs. `Uniform Sampling` vs. `Edge Shadow-Disagreement Trigger (1.2%) + SigLIP Mining + 4D BEV Teacher Auto-Labeling` (`#542`)

- **Chinese FAQ**: [`../../docs/components/enterprise-autonomous-driving-corner-case-mining-4d-autolabel-sizing-faq.md`](../../docs/components/enterprise-autonomous-driving-corner-case-mining-4d-autolabel-sizing-faq.md)
- **English FAQ**: [`../../docs/components/enterprise-autonomous-driving-corner-case-mining-4d-autolabel-sizing-faq.en.md`](../../docs/components/enterprise-autonomous-driving-corner-case-mining-4d-autolabel-sizing-faq.en.md)
- **Interactive H5 Lab**: [`../../docs/h5/autonomous-driving-corner-case-mining-4d-autolabel-lab.html`](../../docs/h5/autonomous-driving-corner-case-mining-4d-autolabel-lab.html)
- **Interactive Sizer**: [`../../docs/reports/enterprise-autonomous-driving-corner-case-mining-4d-autolabel-sizing-sizer.html`](../../docs/reports/enterprise-autonomous-driving-corner-case-mining-4d-autolabel-sizing-sizer.html)
- **7-Card Report**: [`../../docs/reports/enterprise-autonomous-driving-corner-case-mining-4d-autolabel-sizing-cards.html`](../../docs/reports/enterprise-autonomous-driving-corner-case-mining-4d-autolabel-sizing-cards.html)

## Reproduce
```bash
python experiments/enterprise_autonomous_driving_corner_case_mining_4d_autolabel_sizing/run.py
python -m pytest tests/test_enterprise_autonomous_driving_corner_case_mining_4d_autolabel_sizing.py -q
```
