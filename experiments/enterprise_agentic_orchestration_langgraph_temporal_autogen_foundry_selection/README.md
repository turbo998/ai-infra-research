# Enterprise Agentic Workflow Orchestration Selection: `In-Memory` vs. `Foundry/Bedrock Agents` vs. `LangGraph` vs. `Temporal + LangGraph` (`R467`)

Deterministic evaluation suite comparing four enterprise AI Agent orchestration and state-persistence architectures across `1.5M` monthly multi-step workflows (`8` average steps/workflow, `25%` requiring asynchronous Human-in-the-Loop `HITL` approvals):

1. **Tier 1**: Stateless In-Memory ReAct Loop (`100.0%` pod-restart workflow loss, `18.4%` duplicate tool side-effects, `42.0%` idle HITL compute waste, `$142,000/mo` TCO)
2. **Tier 2**: Fully Managed Cloud Agent PaaS (`Azure AI Foundry Agents / AWS Bedrock Agents`) (`1.2%` restart loss, `4.5%` duplicate side-effects, `78.0%` replay recovery, `$86,000/mo` TCO)
3. **Tier 3**: Stateful Graph Framework + DB Checkpointer (`LangGraph / Semantic Kernel + Postgres`) (`0.8%` restart loss, `2.1%` duplicate side-effects, `94.0%` replay recovery, `$58,000/mo` TCO)
4. **Tier 4**: Durable Execution Engine + Graph Cognitive Layer (`Temporal / Azure Durable Functions + LangGraph + Idempotency Keys`) (`0.0%` restart loss, `0.0%` duplicate side-effects, `0.0%` idle HITL compute waste, `100.0%` replay recovery, `$39,000/mo` TCO)

## Usage

```bash
python experiments/enterprise_agentic_orchestration_langgraph_temporal_autogen_foundry_selection/run.py
python experiments/enterprise_agentic_orchestration_langgraph_temporal_autogen_foundry_selection/run.py --check
python -m unittest tests/test_enterprise_agentic_orchestration_langgraph_temporal_autogen_foundry_selection.py
```
