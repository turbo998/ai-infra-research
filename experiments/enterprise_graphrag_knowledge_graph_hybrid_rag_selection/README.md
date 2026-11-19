# Enterprise `GraphRAG` vs. Hybrid Vector/BM25 RAG Architecture & TCO Inflection-Point Selection (`R483`)

Deterministic evaluation suite comparing four enterprise RAG and Knowledge Graph (`GraphRAG`) architectures across a `1,000,000-document` (`20M` chunk) enterprise corpus (`80%` single-hop factual lookups, `20%` multi-hop relational & global thematic queries):

1. **Tier 1**: Naive Dense Vector RAG Only (`81.5%` single-hop accuracy, `34.0%` multi-hop accuracy, `$450` index cost, `$88,000/mo` TCO incl. rework)
2. **Tier 2**: Standard Hybrid RAG (`Dense Vector + BM25 + RRF + Reranker`) (`95.8%` single-hop accuracy, `58.5%` multi-hop accuracy, `$1,200` index cost, `$42,000/mo` TCO)
3. **Tier 3**: 100% Full-Corpus Frontier-LLM `GraphRAG` (`Entity/Edge + Leiden Community Summaries` on All 20M Chunks) (`93.2%` single-hop, `94.6%` multi-hop, `$148,000` upfront index build cost, `4,850ms` P95 latency, `$215,000/mo` TCO)
4. **Tier 4**: Query-Routed Hybrid `BM25+Vector` (80% Queries) + `LazyGraphRAG / LightRAG` + Deterministic Metadata/`GLiNER` Graph (20% Multi-Hop) (`96.8%` single-hop, `96.2%` multi-hop, `$3,800` index build cost, `820ms` P95 latency, `$29,500/mo` TCO)

## Usage

```bash
python experiments/enterprise_graphrag_knowledge_graph_hybrid_rag_selection/run.py
python experiments/enterprise_graphrag_knowledge_graph_hybrid_rag_selection/run.py --check
python -m unittest tests/test_enterprise_graphrag_knowledge_graph_hybrid_rag_selection.py
```
