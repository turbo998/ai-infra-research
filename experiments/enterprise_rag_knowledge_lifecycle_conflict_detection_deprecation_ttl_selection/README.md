# Enterprise RAG Knowledge Lifecycle, NLI Contradiction Detection & Version Supersession Selection Benchmark

Deterministic quantitative evaluation model for [`docs/components/enterprise-rag-knowledge-lifecycle-conflict-detection-deprecation-ttl-selection-faq.md`](../../docs/components/enterprise-rag-knowledge-lifecycle-conflict-detection-deprecation-ttl-selection-faq.md) and [`.en.md`](../../docs/components/enterprise-rag-knowledge-lifecycle-conflict-detection-deprecation-ttl-selection-faq.en.md).

## Provenance
- **Label**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)

## Run & Verify
```bash
python experiments/enterprise_rag_knowledge_lifecycle_conflict_detection_deprecation_ttl_selection/run.py
python experiments/enterprise_rag_knowledge_lifecycle_conflict_detection_deprecation_ttl_selection/run.py --check
python -m unittest tests/test_enterprise_rag_knowledge_lifecycle_conflict_detection_deprecation_ttl_selection.py
```
