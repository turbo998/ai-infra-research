# Enterprise Text-to-SQL, Semantic Layer & BI Data Agent Architecture Evaluator (Round 378)

Deterministic analytical evaluator comparing four enterprise ChatBI / Text-to-SQL architectures on a 300-table cloud data warehouse (`500,000` queries/month):

1. **Tier 1 (`tier1_naive_full_ddl_prompt`)**: Naive Full-DDL Prompt Stuffing (`52.0%` business metric accuracy, `9.8 s` P95 latency, `4.6%` runaway full-scan incident rate, `$38,500/mo` LLM + warehouse compute cost).
2. **Tier 2 (`tier2_dynamic_schema_rag_fewshot`)**: Dynamic Schema-RAG + Golden SQL Few-Shot (`73.5%` business metric accuracy, `4.2 s` P95 latency, `1.4%` runaway scan rate, `$14,200/mo`, suffers from cross-department metric formula drift).
3. **Tier 3 (`tier3_semantic_layer_metric_api`)**: Governed Semantic Layer / Metric API First (`dbt MetricFlow / Cube / Microsoft Fabric IQ / LookML`, `92.8%` business metric accuracy, `2.1 s` P95 latency, `0.1%` runaway scan rate, `$7,600/mo`).
4. **Tier 4 (`tier4_semantic_plus_sqlglot_rls_self_heal`)**: Hybrid Semantic Layer (`85%` governed KPIs) + Sandboxed Text-to-SQL (`15%` ad-hoc queries) + SQLGlot AST Gate + `EXPLAIN` Byte-Scan Circuit Breaker (`10 GB` cap) + Execution Self-Healing + Native Warehouse OAuth OBO RLS/CLS (`96.4%` business metric accuracy, `2.3 s` P95 latency, `0.0%` runaway scan rate, `$8,200/mo`, `-78.7%` vs. Tier 1).

## Usage

```bash
python experiments/enterprise_text_to_sql_semantic_layer_bi_agent_selection/run.py
python experiments/enterprise_text_to_sql_semantic_layer_bi_agent_selection/run.py --check
```
