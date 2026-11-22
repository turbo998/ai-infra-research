# Enterprise Test-Time Compute Scaling (`Always-On Long-CoT` vs. `Best-of-N PRM` vs. `3-Band Adaptive Thinking Budget Router`) (`#482`)

Reproducible analytical model comparing Always-On Unbounded Long-CoT (`64 GPUs`, `$9.80/1k queries`), Parallel `Best-of-N (N=8)` + PRM (`14.5s` latency, `72 GPUs`), Non-Thinking Instruct (`68.5%` accuracy), and a `3-Band Complexity Router + Adaptive Thinking Budget` (`16 GPUs`, `93.9%` accuracy, `-82.0%` cost).

- **Provenance**: `synthetic + anonymized_reported (independently_reproduced=false)`
- **Run**: `python experiments/enterprise_test_time_compute_reasoning_router_thinking_budget_sizing_selection/run.py --check`
