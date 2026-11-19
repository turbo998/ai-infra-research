# Enterprise `Text-to-SQL`, Semantic Metric Layer & Governed Data Analytics Agent Selection (`R482`)

Deterministic evaluation suite comparing four enterprise natural-language data analytics (`Text-to-SQL` / `Chat-with-Data`) architectures across a `650-table` enterprise data warehouse (`50 Billion` fact rows, `180` governed business KPIs, `1,800,000` monthly analytical queries):

1. **Tier 1**: Raw Physical Database `CREATE TABLE` DDL Dump into Prompt (`44.5%` KPI accuracy, `38.0%` metric consistency, `28.5%` runaway full-table scan risk, `$174,000/mo` warehouse + rework TCO)
2. **Tier 2**: Vector Schema RAG + Golden SQL Few-Shot Examples on Physical Tables (`76.2%` accuracy, `68.5%` consistency, `14.2%` runaway scan risk, `$118,000/mo` TCO)
3. **Tier 3**: Domain Fine-Tuned SQL LLM (`SQLCoder / 70B LoRA`) on Physical Tables (`81.8%` accuracy, `74.0%` consistency, `9.5%` runaway scan risk, `$96,000/mo` TCO)
4. **Tier 4**: Governed Semantic Metric Layer (`dbt Semantic Layer / Cube / Microsoft Fabric IQ / Snowflake Cortex`) + Cell-Value Entity Index + `SQLGlot` AST Cost/RLS Gate (`97.4%` KPI accuracy, `100.0%` metric consistency, `0.0%` runaway scan risk, `100.0%` RLS compliance, `$33,000/mo` TCO)

## Usage

```bash
python experiments/enterprise_text2sql_semantic_layer_governed_bi_agent_selection/run.py
python experiments/enterprise_text2sql_semantic_layer_governed_bi_agent_selection/run.py --check
python -m unittest tests/test_enterprise_text2sql_semantic_layer_governed_bi_agent_selection.py
```
