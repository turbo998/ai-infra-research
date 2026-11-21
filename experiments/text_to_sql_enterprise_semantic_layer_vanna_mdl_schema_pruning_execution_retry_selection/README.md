# Enterprise Text-to-SQL & Semantic Layer Selection Experiment (`Semantic MDL + Column/Value Pruning + SQLGlot AST Gate` vs. `Raw DDL Dump` vs. `Vector DDL RAG`)

Reproducible accuracy, latency, and data-warehouse TCO model comparing 4 Text-to-SQL architectures on an `800`-table (`18,000`-column) enterprise data warehouse across `450,000` monthly BI queries.

## Run
```bash
python experiments/text_to_sql_enterprise_semantic_layer_vanna_mdl_schema_pruning_execution_retry_selection/run.py
python experiments/text_to_sql_enterprise_semantic_layer_vanna_mdl_schema_pruning_execution_retry_selection/run.py --check
```
