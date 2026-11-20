# Enterprise Long-Context & Multi-Turn Agent Context Compression & Prefix-Cache Preservation Benchmark

Deterministic quantitative evaluation model for [`docs/components/enterprise-llm-context-compression-history-pruning-prefix-cache-budget-selection-faq.md`](../../docs/components/enterprise-llm-context-compression-history-pruning-prefix-cache-budget-selection-faq.md) and [`.en.md`](../../docs/components/enterprise-llm-context-compression-history-pruning-prefix-cache-budget-selection-faq.en.md).

## Provenance
- **Label**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)

## Run & Verify
```bash
python experiments/enterprise_llm_context_compression_history_pruning_prefix_cache_budget_selection/run.py
python experiments/enterprise_llm_context_compression_history_pruning_prefix_cache_budget_selection/run.py --check
python -m unittest tests/test_enterprise_llm_context_compression_history_pruning_prefix_cache_budget_selection.py
```
