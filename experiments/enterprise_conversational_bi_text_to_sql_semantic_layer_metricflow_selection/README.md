# Enterprise Conversational BI: Raw Text-to-SQL vs. Governed Semantic Layer (`Cube / MetricFlow`) Benchmark

Deterministic quantitative evaluation model for [`docs/components/enterprise-conversational-bi-text-to-sql-semantic-layer-metricflow-selection-faq.md`](../../docs/components/enterprise-conversational-bi-text-to-sql-semantic-layer-metricflow-selection-faq.md) and [`.en.md`](../../docs/components/enterprise-conversational-bi-text-to-sql-semantic-layer-metricflow-selection-faq.en.md).

## Provenance
- **Label**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)

## Run & Verify
```bash
python experiments/enterprise_conversational_bi_text_to_sql_semantic_layer_metricflow_selection/run.py
python experiments/enterprise_conversational_bi_text_to_sql_semantic_layer_metricflow_selection/run.py --check
python -m unittest tests/test_enterprise_conversational_bi_text_to_sql_semantic_layer_metricflow_selection.py
```
