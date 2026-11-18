# Enterprise Long-Context (128K–1M) vs. Hierarchical Full-Section RAG & KV Cache Evaluator 

Deterministic analytical evaluator comparing four enterprise long-document (`150,000 tokens`, 300 pages) architectures at `1,000,000` queries/month:

1. **Tier 1 (`tier1_bruteforce_150k_full_context_stuffing`)**: Brute-Force 150K Full-Document Context Stuffing (`74.2%` multi-hop accuracy due to Lost-in-the-Middle attention dilution, `11,400 ms` P95 TTFT, `2` max concurrent users per 80GB GPU, `$94,000/mo`).
2. **Tier 2 (`tier2_naive_top5_short_chunk_rag`)**: Naive Top-5 Short-Chunk RAG (`512-token` chunks, `76.5%` multi-hop accuracy, `680 ms` P95 TTFT, `64` concurrent users/GPU, `$4,200/mo`).
3. **Tier 3 (`tier3_prefix_pinned_150k_long_context`)**: Prefix-Cached 150K Long Context (`86.0%` accuracy, `1,850 ms` P95 TTFT, `12` concurrent users/GPU, `$19,500/mo` — only works for a small number of static playbooks).
4. **Tier 4 (`tier4_hierarchical_tree_16k_full_section_edge_reorder`)**: Hierarchical Document Tree Routing (`PageIndex / ToC Summary Tree`) + `16K` Full-Section Assembly + Lost-in-the-Middle Edge Reordering + Hot-Section KV Pool (`96.2%` multi-hop accuracy [`+22.0 pts` vs. Tier 1, `+19.7 pts` vs. Tier 2], `540 ms` P95 TTFT [`21.1x` faster than Tier 1], `48` concurrent users/GPU, `$7,800/mo` [`-91.7%` vs. Tier 1]).

## Usage

```bash
python experiments/enterprise_long_context_128k_1m_vs_rag_kv_cache_selection/run.py
python experiments/enterprise_long_context_128k_1m_vs_rag_kv_cache_selection/run.py --check
```
