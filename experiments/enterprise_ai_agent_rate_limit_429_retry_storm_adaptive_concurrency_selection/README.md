# Enterprise Multi-Agent `HTTP 429` Retry-Storm Suppression & Adaptive Concurrency Selection (`R510`)

Deterministic quantitative evaluation model comparing four enterprise multi-agent rate-limit and concurrency-control architectures under **2x traffic surges and transient quota brownouts**:
- **Tier 1**: Uncoordinated Multi-Layer Retries (`App SDK 3x` × `Orchestrator 3x` × `Gateway 3x` = `27x` Amplification Storm)
- **Tier 2**: Client-Side Exponential Backoff with Full Jitter Only (`max_retries=5` at SDK Layer, Ignoring `Retry-After-Ms`)
- **Tier 3**: Static Hard Concurrency Semaphore (`max_concurrent=50`) Without Token-Weight Estimation or Priority Shedding
- **Tier 4**: Single-Layer Gateway Retry (`max_retries=0` in SDKs) + `10%` Global Retry Budget + `x-ratelimit-remaining-tokens / Retry-After-Ms` Pacing + `Gradient2` Adaptive Concurrency

## Verification

```bash
python experiments/enterprise_ai_agent_rate_limit_429_retry_storm_adaptive_concurrency_selection/run.py --check
python -m unittest tests/test_enterprise_ai_agent_rate_limit_429_retry_storm_adaptive_concurrency_selection.py
```
