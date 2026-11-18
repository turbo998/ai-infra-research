# Enterprise Prompt Caching, Semantic Caching & KV-Aware Gateway Routing Evaluator 

Deterministic analytical evaluator comparing four enterprise caching and gateway routing architectures across an 8-replica cluster (`15,000,000` requests/month, `16,000` avg prompt tokens):

1. **Tier 1 (`tier1_dynamic_prompt_round_robin`)**: Dynamic Timestamp-at-Top Prompt + Round-Robin Gateway (`18.5%` effective cache hit rate, `1,420 ms` P95 TTFT, `$68,000/mo`).
2. **Tier 2 (`tier2_naive_vector_semantic_cache`)**: Naive Vector Semantic Cache at Gateway (`cosine >= 0.90`, `42.0%` hit rate, `4.8%` catastrophic false-positive wrong answer rate on numbers/years/SKUs, `680 ms` P95 TTFT, `$41,500/mo`).
3. **Tier 3 (`tier3_static_first_prompt_prefix_cache`)**: Static-First Prompt Engineering + Cloud/Local Exact Prefix Cache with Round-Robin (`58.0%` hit rate, `0.0%` false positives, `510 ms` P95 TTFT, `$31,200/mo`).
4. **Tier 4 (`tier4_kv_aware_router_plus_verified_cache`)**: Static-First Prompt + KV-Aware Prefix-Tree Gateway Router + Entity-Verified Tenant Semantic Cache + Tiered Host-DRAM KV Offload (`89.4%` effective cache hit rate, `0.0%` false positives, `210 ms` P95 TTFT [`6.76x` faster than Tier 1], `$18,400/mo` [`-72.9%` savings]).

## Usage

```bash
python experiments/enterprise_prompt_caching_semantic_cache_kv_aware_routing_selection/run.py
python experiments/enterprise_prompt_caching_semantic_cache_kv_aware_routing_selection/run.py --check
```
