# Enterprise Autonomous DBA & Cloud Database Copilot: Slow-SQL Diagnosis, HypoPG Virtual-Index & Online DDL Evaluator

Deterministic evaluator comparing four Autonomous DBA and Cloud Database Copilot architectures across 450 production PostgreSQL/MySQL OLTP instances, 18,000 monthly slow-SQL regressions, and 1,200 TB of transactional storage:
- **Tier 1**: Manual DBA Firefighting + Ad-Hoc Heuristic Indexing (`58.0%` P99 latency reduction, `2.4%` lock outage / regression rate, `145.0 min` MTTR, `$240,000/mo`)
- **Tier 2**: Prompt-Only LLM Reading Slow SQL & Directly Running `CREATE INDEX` on Production (`52.5%` P99 reduction, `16.5%` MDL lock outage / plan regression rate, `15.0 min` MTTR, `$385,000/mo`)
- **Tier 3**: Standard `EXPLAIN` Plan Text RAG + Static SQL Rewrite Advisor (`74.0%` P99 reduction, `4.6%` regression rate, `52.0 min` MTTR, `$118,000/mo`)
- **Tier 4**: `HypoPG` Virtual-Index Cost Verifier + `SQLGlot` AST Equivalence Rewriter + Non-Blocking Online DDL (`CONCURRENTLY / gh-ost + lock_timeout=2s`) Gate (`94.2%` P99 reduction, `0.0%` lock outage / regression rate, `5.8 min` MTTR, `$36,000/mo`)

## Usage
```bash
python experiments/enterprise_autonomous_dba_slow_sql_hypopg_online_ddl_agent_selection/run.py
python experiments/enterprise_autonomous_dba_slow_sql_hypopg_online_ddl_agent_selection/run.py --check
```
