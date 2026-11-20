# Enterprise Agent Long-Term Memory, Bi-Temporal Fact Upsert & Memory-Poisoning Firewall Benchmark

Deterministic quantitative evaluation model for [`docs/components/enterprise-agent-long-term-memory-consolidation-conflict-upsert-gdpr-selection-faq.md`](../../docs/components/enterprise-agent-long-term-memory-consolidation-conflict-upsert-gdpr-selection-faq.md) and [`.en.md`](../../docs/components/enterprise-agent-long-term-memory-consolidation-conflict-upsert-gdpr-selection-faq.en.md).

## Provenance
- **Label**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)

## Run & Verify
```bash
python experiments/enterprise_agent_long_term_memory_consolidation_conflict_upsert_gdpr_selection/run.py
python experiments/enterprise_agent_long_term_memory_consolidation_conflict_upsert_gdpr_selection/run.py --check
python -m unittest tests/test_enterprise_agent_long_term_memory_consolidation_conflict_upsert_gdpr_selection.py
```
