# Semantic Caching vs. Exact Prefix KV Cache Selection Experiment (`RedisVL / GPTCache` vs. `vLLM / SGLang Prefix KV Cache` & Entity-Guarded Hybrid Caching)

Reproducible false-positive collision rate, compute FLOP savings, average latency, and monthly cluster TCO model comparing 4 caching architectures across `300 RPS` mixed enterprise FAQ and personalized Agent/RAG workloads.

## Run
```bash
python experiments/semantic_caching_redis_gptcache_exact_prefix_kv_cache_false_positive_selection/run.py
python experiments/semantic_caching_redis_gptcache_exact_prefix_kv_cache_false_positive_selection/run.py --check
```
