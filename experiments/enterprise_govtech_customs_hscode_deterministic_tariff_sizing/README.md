# Enterprise GovTech Customs `HS-Code` Classification & Deterministic FTA Tariff Engine (`20M Declarations/Month`): `Manual Broker` vs. `Unconstrained 70B` vs. `Hierarchical HS-Code Tree + 8B/70B + Deterministic FTA Tariff Engine` (`#562`)

- **Chinese FAQ**: [`../../docs/components/enterprise-govtech-customs-hscode-deterministic-tariff-sizing-faq.md`](../../docs/components/enterprise-govtech-customs-hscode-deterministic-tariff-sizing-faq.md)
- **English FAQ**: [`../../docs/components/enterprise-govtech-customs-hscode-deterministic-tariff-sizing-faq.en.md`](../../docs/components/enterprise-govtech-customs-hscode-deterministic-tariff-sizing-faq.en.md)
- **Interactive H5 Lab**: [`../../docs/h5/govtech-customs-hscode-deterministic-tariff-lab.html`](../../docs/h5/govtech-customs-hscode-deterministic-tariff-lab.html)
- **Interactive Sizer**: [`../../docs/reports/enterprise-govtech-customs-hscode-deterministic-tariff-sizing-sizer.html`](../../docs/reports/enterprise-govtech-customs-hscode-deterministic-tariff-sizing-sizer.html)
- **7-Card Report**: [`../../docs/reports/enterprise-govtech-customs-hscode-deterministic-tariff-sizing-cards.html`](../../docs/reports/enterprise-govtech-customs-hscode-deterministic-tariff-sizing-cards.html)

## Reproduce
```bash
python experiments/enterprise_govtech_customs_hscode_deterministic_tariff_sizing/run.py
python -m pytest tests/test_enterprise_govtech_customs_hscode_deterministic_tariff_sizing.py -q
```
