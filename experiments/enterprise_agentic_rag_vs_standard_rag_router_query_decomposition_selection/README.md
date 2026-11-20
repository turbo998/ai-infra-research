# Enterprise Agentic RAG vs. Single-Hop Hybrid RAG Selection & Query-Complexity Routing (`R497`)

Deterministic quantitative evaluation model comparing four enterprise RAG architectures across **30M monthly enterprise queries (`75%` single-hop lookups + `25%` multi-hop comparative/cross-source queries)**:
- **Tier 1**: Single-Hop Naive Vector RAG for All Enterprise Queries
- **Tier 2**: Unbounded Autonomous ReAct `Agentic RAG` Loop on 100% of Queries (`3–6 Serial Hops`)
- **Tier 3**: Static 2-Hop Query Rewrite + Sequential Multi-Query RAG (No Complexity Router or Parallel Sub-Queries)
- **Tier 4**: `12ms` Query-Complexity Router (`75% Fast Single-Hop Hybrid RAG` vs. `25% Bounded 2-Step Parallel DAG Agentic RAG` + `CRAG` Relevance Gate)

## Verification

```bash
python experiments/enterprise_agentic_rag_vs_standard_rag_router_query_decomposition_selection/run.py --check
python -m unittest tests/test_enterprise_agentic_rag_vs_standard_rag_router_query_decomposition_selection.py
```
