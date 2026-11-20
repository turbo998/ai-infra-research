# Enterprise LLM Load Testing, Streaming SSE Benchmarking (`GenAI-Perf / GuideLLM / Locust`) & KV-Cache Knee Selection (`R499`)

Deterministic quantitative evaluation model comparing four enterprise LLM load-testing and capacity-acceptance methodologies:
- **Tier 1**: Legacy HTTP Load Tester (`JMeter / wrk / ab`) with Identical 50-Token Prompt & Non-Streaming `stream=false`
- **Tier 2**: Synthetic Random-Token Streaming Benchmark (`100% Cache-Miss` Uniform Length `4,096 In / 512 Out`)
- **Tier 3**: Open-Loop Average-Latency Benchmark (Ignoring `Goodput SLO: P99 TTFT <= 800ms & P95 TPOT <= 25ms`)
- **Tier 4**: `NVIDIA GenAI-Perf / GuideLLM + Locust SSE` with Production Log-Normal Token Distribution, Realistic Prefix-Cache Tree & `Goodput SLO` Knee Finder

## Verification

```bash
python experiments/enterprise_llm_load_testing_locust_genai_perf_capacity_benchmark_selection/run.py --check
python -m unittest tests/test_enterprise_llm_load_testing_locust_genai_perf_capacity_benchmark_selection.py
```
