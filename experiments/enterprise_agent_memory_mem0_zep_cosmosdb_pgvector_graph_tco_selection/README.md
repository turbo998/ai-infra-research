# Enterprise Agentic Memory & Cross-Session Personalization Selection (`Mem0 / Zep` vs. `Letta` vs. `Azure Cosmos DB` vs. `PostgreSQL pgvector + AGE`)

Reproducible analytical model comparing four enterprise long-term agent memory architectures across 10M users and 50M sessions/month (`memory tokens per turn`, `temporal state update accuracy`, `retrieval latency`, and `monthly TCO`).

- **Provenance**: `synthetic + anonymized_reported (independently_reproduced=false)`
- **Run**: `python experiments/enterprise_agent_memory_mem0_zep_cosmosdb_pgvector_graph_tco_selection/run.py`
