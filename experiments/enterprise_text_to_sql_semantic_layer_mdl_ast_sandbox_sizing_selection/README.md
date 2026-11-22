# Enterprise Text-to-SQL & Conversational BI (`Semantic Metric Layer MDL` + `FK-Subgraph Linking` + `AST/EXPLAIN Gate`) Sizer (`#499`)

Reproducible analytical model comparing 4 enterprise Text-to-SQL / Conversational BI architectures over a `500-table`, `12,000-column` enterprise data warehouse at `30 QPS`:
- **Tier 1 (`Raw 500-Table DDL Stuffing + 70B LLM`)**: `95,000` prompt tokens, `38.5%` business SQL accuracy, `6,800ms` latency, `$42,000/month` in runaway full-table warehouse scans, and **`48x H200 GPUs`**.
- **Tier 2 (`Vector Top-15 Raw DDL Retrieval + 70B LLM`)**: `64.2%` accuracy (metric/join drift), `12x H200 GPUs`.
- **Tier 3 (`Multi-Turn Self-Debug SQL Agent with 5 Live DB Retries`)**: `86.5%` accuracy, `5,400ms` latency, `20x H200 GPUs`.
- **Tier 4 (`Semantic Metric Layer MDL + FK-Subgraph Schema Linking + sqlglot AST/EXPLAIN Gate + 32B/14B SQL Specialist`)**: Raises business SQL accuracy to **`95.4%` (`+56.9 pts`)**, cuts prompt tokens by **`97.5%` (`2,400` vs. `95,000`)**, reduces `P95` latency to **`620ms` (`11.0x` faster)**, eliminates runaway warehouse scans (`$0`), and runs `30 QPS` on just **`4x H200 GPUs` (`-91.7%` GPUs)**.

## Run
```bash
python experiments/enterprise_text_to_sql_semantic_layer_mdl_ast_sandbox_sizing_selection/run.py
```
