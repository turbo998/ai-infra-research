# SGLang Uncached-Token Prefill Bucket Routing & Idle Spill, Mooncake Bounded In-Flight Weight Resharding & LMCache APC-Excluded Hit Metrics

Deterministic Python 3.12 standard-library model (`own-offline`) verifying:
1. `sgl-project/sglang@df13d5dc` (`#45103`): `--route-by-uncached-tokens` matching prefill buckets on `E = L - reusable_tokens`, cold context-admitted union routing, and `CacheAwareDecision::IdleSpill` when all target DP ranks have pending prefill (`waiting_reqs + waiting_uncached_tokens > 0`).
2. `kvcache-ai/Mooncake@0977fc70` (`#4628`): Bounded in-flight weight transfer execution (`max_inflight_batches=N`) via `submit_scatter_read/write`, shared monotonic deadline across `_CompositeCompletionTicket`, and `_UndrainableCompletionUnknownTicket(owner)` lifetime quarantine.
3. `LMCache/LMCache@168130df` (`#5836`): Excluding `apc_covered_tokens = start_chunk * chunk_size` from `requested_tokens` and `l1_hit_tokens`, while reporting `request_hit_tokens = max(engine_hit_tokens, apc_covered_tokens + hit_tokens)` over `prompt_tokens`.
