# Enterprise Long-Horizon Agentic Memory (`Mem0 / Zep Bi-Temporal Graph` vs. `Vector History` vs. `128K Replay`) Sizer (`#501`)

Reproducible analytical model comparing 4 long-horizon agent memory architectures across `100,000 active enterprise users` (`50 RPS`, 6-month interaction history):
- **Tier 1 (`128K Full-History Replay`)**: `128,000` tokens/turn, `62.0%` temporal state accuracy (`Lost-in-the-Middle` contradictions), `3,450ms P95 TTFT`, requires **`80x H200 GPUs`** (`$230,000/month`).
- **Tier 2 (`Rolling Recursive Summary Only`)**: `68.5%` accuracy (loses fine-grained numbers/entities across repeated compression).
- **Tier 3 (`Pure Vector Top-10 Message RAG`)**: `54.5%` temporal state accuracy (retrieves superseded stale facts alongside new facts with equal cosine similarity).
- **Tier 4 (`3-Tier Bi-Temporal Memory: 6-Turn Working Buffer + Async 8B Extractor + Mem0/Zep Bi-Temporal Fact Graph`)**: Raises temporal state accuracy to **`94.8%` (`+40.3 pts` over Vector RAG)**, cuts prompt tokens by **`98.6%` (`1,850` vs. `128,000`)**, achieves **`195ms P95 TTFT`**, and serves `50 RPS` on just **`4x H200 GPUs` (`-95.0%` GPUs)**.

## Run
```bash
python experiments/enterprise_agentic_memory_mem0_zep_bitemporal_graph_kv_sizing_selection/run.py
```
