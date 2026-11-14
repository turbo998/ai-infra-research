# Enterprise LLM Context Caching, Prompt-Cache Billing Economics & KV TTL Pinning Selection (`llm_context_caching_prompt_cache_billing_ttl_selection`)

Deterministic Python 3.12 analytical evaluator for **Round 173**: comparing `Stateless Full Recompute` (Tier 1), `Naive Round-Robin LB + Local HBM LRU Cache` (Tier 2), `Canonical Static-First Prompt + TTL Pinning + Host Pinned DRAM L2` (Tier 3), and `Global KV-Aware Radix Router + Distributed RDMA KV Pool + Append-Only Agent History` (Tier 4).

## Reproduction

```bash
python -m experiments.llm_context_caching_prompt_cache_billing_ttl_selection.run
python -m unittest tests/test_llm_context_caching_prompt_cache_billing_ttl_selection.py -q
```

## Provenance

- `type`: `synthetic_analytical_model_calibrated_to_published_caching_benchmarks`
- `independently_reproduced`: `false`
- `physical_gpu_cluster_measurement`: `not_measured`
