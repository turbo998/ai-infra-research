# Enterprise Durable Agentic Workflows (`Temporal + LangGraph Checkpointing + HITL`) Sizer (`#527`)

Reproducible analytical model comparing **In-Memory Stateless Agent Loops (`71.2%` 20-step completion, `8.4%` duplicate tool side-effects, `44%` wasted recompute tokens, `24 GPUs`)**, **Managed `Azure AI Foundry Agent Service + Durable Functions`**, and **`Temporal + LangGraph Postgres Checkpointer + Idempotency Keys + Tiered KV Offload` (`99.7%` completion, `0.0%` duplicate side effects, `12x H200 GPUs`, `-50.0%` GPUs 🏆)**.

## Run

```bash
python experiments/enterprise_durable_agent_workflow_temporal_langgraph_hitl_sizing_selection/run.py
```

- **Provenance**: `synthetic + anonymized_reported (independently_reproduced=false)`
- **Bilingual Guides**:
  - [Chinese FAQ](../../docs/components/enterprise-durable-agent-workflow-temporal-langgraph-hitl-sizing-selection-faq.md)
  - [English FAQ](../../docs/components/enterprise-durable-agent-workflow-temporal-langgraph-hitl-sizing-selection-faq.en.md)
- **Interactive H5 Lab**: [docs/h5/durable-agent-workflow-temporal-langgraph-hitl-lab.html](../../docs/h5/durable-agent-workflow-temporal-langgraph-hitl-lab.html)
