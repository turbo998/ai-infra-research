# Enterprise Real-Time AI Safety Guardrails (`Speculative Parallel Input Shield` & `16-Token Sliding-Window Output Circuit Breaker`) (`#473`)

Reproducible analytical model comparing Synchronous Full-Response Buffering, Synchronous Blocking Input Guardrails (`+185ms TTFT`), Regex-Only Filters (`41% recall`), and Speculative Parallel Input Shield (`0.8ms` added `TTFT`) + 16-Token Sliding-Window Async Output Circuit Breaker.

- **Provenance**: `synthetic + anonymized_reported (independently_reproduced=false)`
- **Run**: `python experiments/enterprise_ai_safety_guardrails_speculative_parallel_shield_ttft_selection/run.py --check`
