# Enterprise Agentic Memory & Long-Term Personalization Evaluator (Round 394)

Deterministic analytical evaluator comparing four enterprise cross-session agent memory architectures across `5,000,000` turns/month (`100` historical sessions per user over 6 months):

1. **Tier 1 (`tier1_sliding_window_raw_chat_stuffing`)**: Sliding-Window Raw Chat History Stuffing (`54.0%` temporal fact accuracy, `18.2%` stale contradiction rate, `1,650 ms` P95 TTFT, `$42,000/mo`).
2. **Tier 2 (`tier2_recursive_rolling_text_summary`)**: Recursive Rolling Text Summary (`61.5%` temporal fact accuracy due to lossy compression decay of exact IDs/constraints, `14.0%` contradiction rate, `340 ms` P95 TTFT, `$6,200/mo`).
3. **Tier 3 (`tier3_naive_append_only_vector_memory`)**: Naive Append-Only Vector Memory Store (`74.5%` temporal fact accuracy, `26.5%` stale contradiction rate when user preferences/states change over months, `390 ms` P95 TTFT, `$8,400/mo`).
4. **Tier 4 (`tier4_three_tier_bitemporal_conflict_resolved_memory`)**: 3-Tier Cognitive Memory (`Working Session Buffer` + `Bi-Temporal Conflict-Resolved Fact Graph [Mem0 / Zep / Graphiti]` + `Episodic Procedural Memory`, `96.8%` temporal fact accuracy [`+42.8 pts` vs. Tier 1, `+22.3 pts` vs. Tier 3], `0.6%` contradiction rate, `310 ms` P95 TTFT [`5.32x` faster than Tier 1], `$7,400/mo` [`-82.4%` vs. Tier 1]).

## Usage

```bash
python experiments/enterprise_agentic_memory_long_term_personalization_selection/run.py
python experiments/enterprise_agentic_memory_long_term_personalization_selection/run.py --check
```
