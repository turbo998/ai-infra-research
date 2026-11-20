# Enterprise Healthcare Prior Authorization: Da Vinci FHIR, CQL Medical Necessity & Statutory Approve-Only Benchmark

Deterministic quantitative evaluation model for [`docs/components/enterprise-healthcare-prior-authorization-fhir-cql-medical-necessity-selection-faq.md`](../../docs/components/enterprise-healthcare-prior-authorization-fhir-cql-medical-necessity-selection-faq.md) and [`.en.md`](../../docs/components/enterprise-healthcare-prior-authorization-fhir-cql-medical-necessity-selection-faq.en.md).

## Provenance
- **Label**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)

## Run & Verify
```bash
python experiments/enterprise_healthcare_prior_authorization_fhir_cql_medical_necessity_selection/run.py
python experiments/enterprise_healthcare_prior_authorization_fhir_cql_medical_necessity_selection/run.py --check
python -m unittest tests/test_enterprise_healthcare_prior_authorization_fhir_cql_medical_necessity_selection.py
```
