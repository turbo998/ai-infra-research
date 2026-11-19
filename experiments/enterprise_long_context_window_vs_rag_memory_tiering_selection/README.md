# Enterprise Long-Context Window (`128K–1M`) vs. Hybrid RAG vs. Hierarchical Agent Memory Selection (`R488`)

Deterministic evaluation suite comparing four enterprise context management, `RAG`, and long-horizon Agent memory architectures across `25,000,000` monthly turns (`500M-token` enterprise knowledge base, `50-turn` multi-session agent workflows):

1. **Tier 1**: Stuffing `250K–1M` Token Corpus into Context Window on Every Turn (`74.2%` multi-needle accuracy, `6,850ms` P95 TTFT, `$385,000/mo` TCO)
2. **Tier 2**: Naive Sliding-Window Chat Truncation (`Last 10 Turns`) + Flat `512-Token` Chunk RAG (`79.5%` multi-needle accuracy, `31.0%` Turn-50 memory recall, `$82,000/mo` TCO)
3. **Tier 3**: Rolling Prose Summarization + Parent-Document RAG (`91.4%` multi-needle accuracy, `74.5%` Turn-50 recall, `$58,000/mo` TCO)
4. **Tier 4**: Adaptive Context Router (`Prefix-Cached 64K Long-Context` for Single-Doc Analysis + `Parent-Doc Hybrid RAG` for Corpus Search + `3-Tier Bi-Temporal Episodic/Semantic Agent Memory`) (`97.2%` multi-needle accuracy, `98.4%` Turn-50 memory recall, `410ms` P95 TTFT, `$31,500/mo` TCO)

## Usage

```bash
python experiments/enterprise_long_context_window_vs_rag_memory_tiering_selection/run.py
python experiments/enterprise_long_context_window_vs_rag_memory_tiering_selection/run.py --check
python -m unittest tests/test_enterprise_long_context_window_vs_rag_memory_tiering_selection.py
```
