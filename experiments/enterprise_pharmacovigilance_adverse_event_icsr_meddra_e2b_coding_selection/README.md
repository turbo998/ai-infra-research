# Enterprise Pharmacovigilance ICSR Intake: MedDRA v27.1 Pinning, IME Seriousness Triage & E2B(R3) Benchmark

Deterministic quantitative evaluation model for [`docs/components/enterprise-pharmacovigilance-adverse-event-icsr-meddra-e2b-coding-selection-faq.md`](../../docs/components/enterprise-pharmacovigilance-adverse-event-icsr-meddra-e2b-coding-selection-faq.md) and [`.en.md`](../../docs/components/enterprise-pharmacovigilance-adverse-event-icsr-meddra-e2b-coding-selection-faq.en.md).

## Provenance
- **Label**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)

## Run & Verify
```bash
python experiments/enterprise_pharmacovigilance_adverse_event_icsr_meddra_e2b_coding_selection/run.py
python experiments/enterprise_pharmacovigilance_adverse_event_icsr_meddra_e2b_coding_selection/run.py --check
python -m unittest tests/test_enterprise_pharmacovigilance_adverse_event_icsr_meddra_e2b_coding_selection.py
```
