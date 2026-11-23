# Enterprise Insurance Claims Multimodal Fraud Detection & Straight-Through Adjudication (`500K Claims/Month`): `Manual Adjuster` vs. `Monolithic 72B VLM` vs. `C2PA/DINOv2 Fraud Hash + YOLOv11 Damage + Policy Graph 70B STP` (`#546`)

- **Chinese FAQ**: [`../../docs/components/enterprise-insurance-claims-multimodal-fraud-stp-adjudication-sizing-faq.md`](../../docs/components/enterprise-insurance-claims-multimodal-fraud-stp-adjudication-sizing-faq.md)
- **English FAQ**: [`../../docs/components/enterprise-insurance-claims-multimodal-fraud-stp-adjudication-sizing-faq.en.md`](../../docs/components/enterprise-insurance-claims-multimodal-fraud-stp-adjudication-sizing-faq.en.md)
- **Interactive H5 Lab**: [`../../docs/h5/insurance-claims-multimodal-fraud-stp-adjudication-lab.html`](../../docs/h5/insurance-claims-multimodal-fraud-stp-adjudication-lab.html)
- **Interactive Sizer**: [`../../docs/reports/enterprise-insurance-claims-multimodal-fraud-stp-adjudication-sizing-sizer.html`](../../docs/reports/enterprise-insurance-claims-multimodal-fraud-stp-adjudication-sizing-sizer.html)
- **7-Card Report**: [`../../docs/reports/enterprise-insurance-claims-multimodal-fraud-stp-adjudication-sizing-cards.html`](../../docs/reports/enterprise-insurance-claims-multimodal-fraud-stp-adjudication-sizing-cards.html)

## Reproduce
```bash
python experiments/enterprise_insurance_claims_multimodal_fraud_stp_adjudication_sizing/run.py
python -m pytest tests/test_enterprise_insurance_claims_multimodal_fraud_stp_adjudication_sizing.py -q
```
