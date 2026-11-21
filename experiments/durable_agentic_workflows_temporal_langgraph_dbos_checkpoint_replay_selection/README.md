# Durable Agentic Workflows Selection Experiment (`Temporal / Restate / DBOS` + `LangGraph Checkpointer` vs. `In-Memory Loop`)

Reproducible reliability, token-recompute waste, duplicate side-effect risk, and TCO model comparing 4 agent workflow orchestration engines across `500,000` monthly 25-step enterprise workflows.

## Run
```bash
python experiments/durable_agentic_workflows_temporal_langgraph_dbos_checkpoint_replay_selection/run.py
python experiments/durable_agentic_workflows_temporal_langgraph_dbos_checkpoint_replay_selection/run.py --check
```
