# Agentic Context Compaction & Prefix KV Preservation Experiment (`85%-Watermark Structured Checkpoint` vs. `Sliding Window & Observation Masking`)

Reproducible analytical model comparing 4 agentic context management architectures across a `50-turn` autonomous coding/research task (`128K` context limit, `3,500` tokens/turn tool observation):
- **Tier 1 (`Rolling FIFO Sliding-Window Drop` `messages.pop(0)` every turn)**: **`0.0%` Prefix KV Cache hit rate** (`4,200 ms` TTFT/turn, `$14.80/task`, `46.0%` 50-turn task success).
- **Tier 2 (`Unstructured LLM Summarization Every 5 Turns`)**: `72.0%` KV hit rate, `68.5%` task success (loses exact file/line/error details).
- **Tier 3 (`Stale Tool Observation Folding >8 Turns Old`)**: `64.0%` KV hit rate, `82.4%` task success, `$2.95/task`.
- **Tier 4 (`85%-Watermark Append-Only Prefix Compaction + Structured Checkpoint Schema + Scratchpad Files`, Recommended)**: **`91.5%` Prefix KV Cache hit rate**, **`380 ms` average turn TTFT (`11.1x` faster)**, **`93.4%` 50-turn task success**, and **`$1.65/task` (`-88.9%` token cost)**.

## Run & Verify

```bash
python experiments/agentic_context_engineering_compaction_observation_masking_prefix_kv_selection/run.py
python experiments/agentic_context_engineering_compaction_observation_masking_prefix_kv_selection/run.py --check
```
