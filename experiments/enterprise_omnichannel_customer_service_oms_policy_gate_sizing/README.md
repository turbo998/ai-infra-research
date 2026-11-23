# Enterprise Omnichannel Customer Service & Order-Action Agent (`10M Tickets/Month`): `Legacy IVR + Human` vs. `Unconstrained 70B` vs. `8B/70B Cascade + Deterministic OMS/OPA Refund Policy Gate` (`#556`)

- **Chinese FAQ**: [`../../docs/components/enterprise-omnichannel-customer-service-oms-policy-gate-sizing-faq.md`](../../docs/components/enterprise-omnichannel-customer-service-oms-policy-gate-sizing-faq.md)
- **English FAQ**: [`../../docs/components/enterprise-omnichannel-customer-service-oms-policy-gate-sizing-faq.en.md`](../../docs/components/enterprise-omnichannel-customer-service-oms-policy-gate-sizing-faq.en.md)
- **Interactive H5 Lab**: [`../../docs/h5/omnichannel-customer-service-oms-policy-gate-lab.html`](../../docs/h5/omnichannel-customer-service-oms-policy-gate-lab.html)
- **Interactive Sizer**: [`../../docs/reports/enterprise-omnichannel-customer-service-oms-policy-gate-sizing-sizer.html`](../../docs/reports/enterprise-omnichannel-customer-service-oms-policy-gate-sizing-sizer.html)
- **7-Card Report**: [`../../docs/reports/enterprise-omnichannel-customer-service-oms-policy-gate-sizing-cards.html`](../../docs/reports/enterprise-omnichannel-customer-service-oms-policy-gate-sizing-cards.html)

## Reproduce
```bash
python experiments/enterprise_omnichannel_customer_service_oms_policy_gate_sizing/run.py
python -m pytest tests/test_enterprise_omnichannel_customer_service_oms_policy_gate_sizing.py -q
```
