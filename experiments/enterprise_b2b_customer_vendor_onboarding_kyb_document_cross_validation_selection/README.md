# Enterprise B2B Customer & Vendor Onboarding (`KYB` Multi-Doc Cross-Validation & `UBO` Graph) Benchmark

Deterministic quantitative evaluation model for [`docs/components/enterprise-b2b-customer-vendor-onboarding-kyb-document-cross-validation-selection-faq.md`](../../docs/components/enterprise-b2b-customer-vendor-onboarding-kyb-document-cross-validation-selection-faq.md) and [`.en.md`](../../docs/components/enterprise-b2b-customer-vendor-onboarding-kyb-document-cross-validation-selection-faq.en.md).

## Provenance
- **Label**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)

## Run & Verify
```bash
python experiments/enterprise_b2b_customer_vendor_onboarding_kyb_document_cross_validation/run.py
python experiments/enterprise_b2b_customer_vendor_onboarding_kyb_document_cross_validation/run.py --check
python -m unittest tests/test_enterprise_b2b_customer_vendor_onboarding_kyb_document_cross_validation_selection.py
```
