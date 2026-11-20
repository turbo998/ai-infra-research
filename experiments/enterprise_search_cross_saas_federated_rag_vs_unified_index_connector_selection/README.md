# Enterprise Cross-SaaS Workplace Search: Live Federated Fan-Out vs. Unified Connector Index Benchmark

Deterministic quantitative evaluation model for [`docs/components/enterprise-search-cross-saas-federated-rag-vs-unified-index-connector-selection-faq.md`](../../docs/components/enterprise-search-cross-saas-federated-rag-vs-unified-index-connector-selection-faq.md) and [`.en.md`](../../docs/components/enterprise-search-cross-saas-federated-rag-vs-unified-index-connector-selection-faq.en.md).

## Provenance
- **Label**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)

## Run & Verify
```bash
python experiments/enterprise_search_cross_saas_federated_rag_vs_unified_index_connector_selection/run.py
python experiments/enterprise_search_cross_saas_federated_rag_vs_unified_index_connector_selection/run.py --check
python -m unittest tests/test_enterprise_search_cross_saas_federated_rag_vs_unified_index_connector_selection.py
```
