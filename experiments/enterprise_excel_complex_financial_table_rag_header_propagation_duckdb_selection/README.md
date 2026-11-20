# Enterprise Complex Excel & Multi-Header Financial Table RAG (`openpyxl AST` Un-Merging + `DuckDB`) Benchmark

Deterministic quantitative evaluation model for [`docs/components/enterprise-excel-complex-financial-table-rag-header-propagation-duckdb-selection-faq.md`](../../docs/components/enterprise-excel-complex-financial-table-rag-header-propagation-duckdb-selection-faq.md) and [`.en.md`](../../docs/components/enterprise-excel-complex-financial-table-rag-header-propagation-duckdb-selection-faq.en.md).

## Provenance
- **Label**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)

## Run & Verify
```bash
python experiments/enterprise_excel_complex_financial_table_rag_header_propagation_duckdb_selection/run.py
python experiments/enterprise_excel_complex_financial_table_rag_header_propagation_duckdb_selection/run.py --check
python -m unittest tests/test_enterprise_excel_complex_financial_table_rag_header_propagation_duckdb_selection.py
```
