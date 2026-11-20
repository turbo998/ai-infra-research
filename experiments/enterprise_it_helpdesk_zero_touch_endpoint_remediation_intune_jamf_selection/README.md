# Enterprise IT Helpdesk & Zero-Touch Endpoint Remediation Agent (`Intune / Jamf / Entra ID`) Benchmark

Deterministic quantitative evaluation model for [`docs/components/enterprise-it-helpdesk-zero-touch-endpoint-remediation-intune-jamf-selection-faq.md`](../../docs/components/enterprise-it-helpdesk-zero-touch-endpoint-remediation-intune-jamf-selection-faq.md) and [`.en.md`](../../docs/components/enterprise-it-helpdesk-zero-touch-endpoint-remediation-intune-jamf-selection-faq.en.md).

## Provenance
- **Label**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)

## Run & Verify
```bash
python experiments/enterprise_it_helpdesk_zero_touch_endpoint_remediation_intune_jamf_selection/run.py
python experiments/enterprise_it_helpdesk_zero_touch_endpoint_remediation_intune_jamf_selection/run.py --check
python -m unittest tests/test_enterprise_it_helpdesk_zero_touch_endpoint_remediation_intune_jamf_selection.py
```
