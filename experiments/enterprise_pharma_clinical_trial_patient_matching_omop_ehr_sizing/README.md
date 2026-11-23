# Enterprise Pharma Clinical Trial Patient-Protocol Matching (`10M EHR Patients`): `Brute-Force 70B` vs. `ICD-10 SQL Only` vs. `OMOP SQL Filter (94%) + 8B Negation-Aware EHR Feature Store + 70B Criterion Verifier` (`#547`)

- **Chinese FAQ**: [`../../docs/components/enterprise-pharma-clinical-trial-patient-matching-omop-ehr-sizing-faq.md`](../../docs/components/enterprise-pharma-clinical-trial-patient-matching-omop-ehr-sizing-faq.md)
- **English FAQ**: [`../../docs/components/enterprise-pharma-clinical-trial-patient-matching-omop-ehr-sizing-faq.en.md`](../../docs/components/enterprise-pharma-clinical-trial-patient-matching-omop-ehr-sizing-faq.en.md)
- **Interactive H5 Lab**: [`../../docs/h5/pharma-clinical-trial-patient-matching-omop-ehr-lab.html`](../../docs/h5/pharma-clinical-trial-patient-matching-omop-ehr-lab.html)
- **Interactive Sizer**: [`../../docs/reports/enterprise-pharma-clinical-trial-patient-matching-omop-ehr-sizing-sizer.html`](../../docs/reports/enterprise-pharma-clinical-trial-patient-matching-omop-ehr-sizing-sizer.html)
- **7-Card Report**: [`../../docs/reports/enterprise-pharma-clinical-trial-patient-matching-omop-ehr-sizing-cards.html`](../../docs/reports/enterprise-pharma-clinical-trial-patient-matching-omop-ehr-sizing-cards.html)

## Reproduce
```bash
python experiments/enterprise_pharma_clinical_trial_patient_matching_omop_ehr_sizing/run.py
python -m pytest tests/test_enterprise_pharma_clinical_trial_patient_matching_omop_ehr_sizing.py -q
```
