# Enterprise Cold-Chain Logistics & Perishable Spoilage Prevention (`1M Reefer Containers`): `Static FIFO` vs. `Free-Form LLM Routing` vs. `IoT Kinetic Shelf-Life + FEFO CuOpt Routing & Markdown MILP` (`#559`)

- **Chinese FAQ**: [`../../docs/components/enterprise-cold-chain-logistics-fefo-shelf-life-cuopt-milp-sizing-faq.md`](../../docs/components/enterprise-cold-chain-logistics-fefo-shelf-life-cuopt-milp-sizing-faq.md)
- **English FAQ**: [`../../docs/components/enterprise-cold-chain-logistics-fefo-shelf-life-cuopt-milp-sizing-faq.en.md`](../../docs/components/enterprise-cold-chain-logistics-fefo-shelf-life-cuopt-milp-sizing-faq.en.md)
- **Interactive H5 Lab**: [`../../docs/h5/cold-chain-logistics-fefo-shelf-life-cuopt-milp-lab.html`](../../docs/h5/cold-chain-logistics-fefo-shelf-life-cuopt-milp-lab.html)
- **Interactive Sizer**: [`../../docs/reports/enterprise-cold-chain-logistics-fefo-shelf-life-cuopt-milp-sizing-sizer.html`](../../docs/reports/enterprise-cold-chain-logistics-fefo-shelf-life-cuopt-milp-sizing-sizer.html)
- **7-Card Report**: [`../../docs/reports/enterprise-cold-chain-logistics-fefo-shelf-life-cuopt-milp-sizing-cards.html`](../../docs/reports/enterprise-cold-chain-logistics-fefo-shelf-life-cuopt-milp-sizing-cards.html)

## Reproduce
```bash
python experiments/enterprise_cold_chain_logistics_fefo_shelf_life_cuopt_milp_sizing/run.py
python -m pytest tests/test_enterprise_cold_chain_logistics_fefo_shelf_life_cuopt_milp_sizing.py -q
```
