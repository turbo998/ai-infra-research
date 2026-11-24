# Experiment #592: Enterprise Reasoning Model `<think>` Budget Router & Multi-Turn KV Stripping Sizer

Reproduces the `75.6%` output token reduction (`940` vs. `3,850` tokens/req) of the `55/30/15` Adaptive Thinking Effort Router, the `74.2%` multi-turn KV memory reduction (`7,320` vs. `28,400` tokens at turn 6) from stripping completed `<think>` blocks, and the `75.0%` GPU fleet savings (`8× H200` vs. `32× H200` at `50 QPS`) (`docs/components/enterprise-reasoning-model-thinking-budget-router-kv-sizing-faq.md`).
