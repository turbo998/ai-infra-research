# Enterprise RAG Chunking Architecture (`Layout AST + 256->1536 Parent-Child + Breadcrumb Header`) (`R503`)

Deterministic quantitative evaluation model comparing four enterprise RAG document chunking architectures across **20,000,000 monthly ingested pages**:
- **Tier 1**: Naive Fixed-Character Sliding Window (`500 chars, 50 overlap`) Ignoring Tables & Headings
- **Tier 2**: LLM-Driven Proposition / Agentic Semantic Chunking (`Calling 8B/70B LLM on Every Paragraph`)
- **Tier 3**: Embedding-Cosine Breakpoint Semantic Chunking Without Parent-Window Expansion or Table Locking
- **Tier 4**: Layout-AST Structural Chunking + `256-Token Child Search -> 1,536-Token Parent Expansion` + Deterministic Breadcrumb Header Injection (`[Doc | Sec | Table Header]`)

## Verification

```bash
python experiments/enterprise_rag_chunking_strategy_semantic_layout_parent_child_window_selection/run.py --check
python -m unittest tests/test_enterprise_rag_chunking_strategy_semantic_layout_parent_child_window_selection.py
```
