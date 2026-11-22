# Enterprise `GraphRAG / LazyGraphRAG` vs. Hybrid `Vector + BM25 + Reranker` vs. `128K Stuffing` Sizer (`#497`)

Reproducible analytical model comparing 4 enterprise knowledge retrieval architectures on a `500M-token` (`1,000,000 chunks`) enterprise corpus at `50 QPS`:
- **Tier 1 (`128K Long-Context Stuffing`)**: `3,850ms` latency, `$185 / 1k queries`, requires `64x H200 GPUs`.
- **Tier 2 (`Naive Full LLM GraphRAG`)**: `93.8%` global multi-hop accuracy, but **`$4,500` index build cost**, **`14,200ms` Map-Reduce query latency**, **`$420 / 1k queries`**, and **`96x H200 GPUs`**.
- **Tier 3 (`Pure Hybrid Vector + BM25 + BGE-Reranker`)**: `94.5%` factoid accuracy, `420ms` latency, `4x H200 GPUs`, but `58.0%` global multi-hop synthesis accuracy.
- **Tier 4 (`Intent-Routed 85% Hybrid Reranker + 15% LazyGraphRAG / LightRAG`)**: Achieves **`94.8%` factoid & `93.2%` global multi-hop accuracy**, cuts index build cost by **`97.2%` (`$128` vs. `$4,500`)**, holds `P95` latency at **`680ms` (`20.9x` faster)**, and requires only **`6x H200 GPUs` (`-93.8%` GPUs)**.

## Run
```bash
python experiments/enterprise_graphrag_lazygraphrag_hybrid_reranker_sizing_selection/run.py
```
