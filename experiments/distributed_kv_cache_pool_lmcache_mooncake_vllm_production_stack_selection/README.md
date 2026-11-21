# Distributed KV Cache Pool & Cross-Replica Routing Selection Experiment (`LMCache + vLLM Production Stack` vs. `Mooncake` vs. `NVIDIA Dynamo KV Router`)

Reproducible capacity and latency model comparing 4 KV Cache pooling architectures across a `32`-replica `vLLM` cluster serving `32K`-token multi-turn agent & shared RAG workloads.

## Run
```bash
python experiments/distributed_kv_cache_pool_lmcache_mooncake_vllm_production_stack_selection/run.py
python experiments/distributed_kv_cache_pool_lmcache_mooncake_vllm_production_stack_selection/run.py --check
```
