# Enterprise Airline IROPS Disruption Recovery (`50,000 Stranded Passengers`): `Manual OCC` vs. `Pure 70B LLM` vs. `70B NOTAM/Contract Compiler + Column-Generation MILP + 8B Passenger Rebooking` (`#552`)

- **Chinese FAQ**: [`../../docs/components/enterprise-airline-irops-disruption-recovery-crew-milp-sizing-faq.md`](../../docs/components/enterprise-airline-irops-disruption-recovery-crew-milp-sizing-faq.md)
- **English FAQ**: [`../../docs/components/enterprise-airline-irops-disruption-recovery-crew-milp-sizing-faq.en.md`](../../docs/components/enterprise-airline-irops-disruption-recovery-crew-milp-sizing-faq.en.md)
- **Interactive H5 Lab**: [`../../docs/h5/airline-irops-disruption-recovery-crew-milp-lab.html`](../../docs/h5/airline-irops-disruption-recovery-crew-milp-lab.html)
- **Interactive Sizer**: [`../../docs/reports/enterprise-airline-irops-disruption-recovery-crew-milp-sizing-sizer.html`](../../docs/reports/enterprise-airline-irops-disruption-recovery-crew-milp-sizing-sizer.html)
- **7-Card Report**: [`../../docs/reports/enterprise-airline-irops-disruption-recovery-crew-milp-sizing-cards.html`](../../docs/reports/enterprise-airline-irops-disruption-recovery-crew-milp-sizing-cards.html)

## Reproduce
```bash
python experiments/enterprise_airline_irops_disruption_recovery_crew_milp_sizing/run.py
python -m pytest tests/test_enterprise_airline_irops_disruption_recovery_crew_milp_sizing.py -q
```
