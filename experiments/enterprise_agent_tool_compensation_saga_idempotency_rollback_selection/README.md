# Enterprise Transactional Agent `Saga` Compensation, `Idempotency-Key` & Rollback Selection (`R512`)

Deterministic quantitative evaluation model comparing four enterprise transactional agent tool-execution architectures across **400,000 monthly multi-system write transactions**:
- **Tier 1**: Uncoordinated Free-Form Agent Tool Loops (`Fire-and-Forget HTTP POSTs` Without Idempotency or Rollback)
- **Tier 2**: Prompt-Driven "LLM Self-Rollback" ("If Step 3 Fails, Ask the LLM to Call Undo Tools")
- **Tier 3**: Distributed Two-Phase Locking (`2PC XA Transactions`) Across External SaaS & Legacy ERP APIs
- **Tier 4**: Deterministic `Saga` Orchestrator + MCP Gateway `Idempotency-Key` (`SHA256(trace_id + step_id)`) + Paired `[Action, Compensate_Fn]` Registry + Commit-Last External Side-Effects

## Verification

```bash
python experiments/enterprise_agent_tool_compensation_saga_idempotency_rollback_selection/run.py --check
python -m unittest tests/test_enterprise_agent_tool_compensation_saga_idempotency_rollback_selection.py
```
