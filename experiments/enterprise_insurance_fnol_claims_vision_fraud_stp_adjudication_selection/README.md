# Enterprise P&C Insurance FNOL & Claims Adjudication: Part-Segmented Vision, Synthetic Fraud Gate & STP Benchmark

Deterministic quantitative evaluation model for [`docs/components/enterprise-insurance-fnol-claims-vision-fraud-stp-adjudication-selection-faq.md`](../../docs/components/enterprise-insurance-fnol-claims-vision-fraud-stp-adjudication-selection-faq.md) and [`.en.md`](../../docs/components/enterprise-insurance-fnol-claims-vision-fraud-stp-adjudication-selection-faq.en.md).

## Provenance
- **Label**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)

## Run & Verify
```bash
python experiments/enterprise_insurance_fnol_claims_vision_fraud_stp_adjudication_selection/run.py
python experiments/enterprise_insurance_fnol_claims_vision_fraud_stp_adjudication_selection/run.py --check
python -m unittest tests/test_enterprise_insurance_fnol_claims_vision_fraud_stp_adjudication_selection.py
```
