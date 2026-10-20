# SGLang Uncached-Token Prefill Bucket Routing & Idle DP Spill, Mooncake Bounded In-Flight Weight Resharding & LMCache APC-Excluded Hit Metrics

> **中文版**: [sglang-router-uncached-bucket-spill-mooncake-inflight-reshard-lmcache-apc-metrics.md](sglang-router-uncached-bucket-spill-mooncake-inflight-reshard-lmcache-apc-metrics.md)  
> **Evidence Class**: `implementation` + `own-offline`  
> **Experiment**: [`experiments/sglang_uncached_bucket_mooncake_inflight_lmcache_apc/run.py`](../../experiments/sglang_uncached_bucket_mooncake_inflight_lmcache_apc/run.py)

---

## 1. Engineering Problem & Context

In disaggregated prefill routing, online weight resharding, and hierarchical KV caching (Engine APC + LMCache L1/L2), three subtle bottlenecks distort scheduling and observability:
1. **SGLang Router Input-Length Bucket Mismatch on Warm Prefixes (`sgl-project/sglang@df13d5dc`, `#45103`)**: A `20,000`-token prompt with `15/20` cached prefix blocks (`15,000` reusable tokens) has only `E = 5,000` uncached extend tokens. Routing by full input length sends it to the `long` bucket (`min_extend_tokens: 8193`), starving `short` buckets (`max_extend_tokens: 8192`) and failing to spill when all target DP ranks are busy.
2. **Mooncake Sequential Batch Draining & Native Ticket Destructor GIL Lock (`kvcache-ai/Mooncake@0977fc70`, `#4628`)**: Draining every scatter weight batch before submitting the next (`max_inflight_batches=1`) serializes submission and DMA wait times, while dropping an unresolved native `ScatterTransferTicket` invokes a C++ destructor that blocks for physical completion with the Python GIL held.
3. **LMCache Inflated L1 Hit Rate from Engine APC Prefix (`LMCache/LMCache@168130df`, `#5836`)**: When the engine's Automatic Prefix Cache (APC) already holds `start_chunk` chunks (`key.start > 0`), adding `job.start_chunk` to `l1_found_count` falsely credited the engine's prefix cache hits to LMCache L1.

---

## 2. First-Hand Code Analysis & Quantitative Results

### 2.1 SGLang `--route-by-uncached-tokens` and `CacheAwareDecision::IdleSpill`
In `experimental/sgl-router/src/policies/{selection.rs,buckets.rs}` (`df13d5dc`):
- **Uncached Extend Work $E$**: Computes `E = request_sequence_tokens.saturating_sub(best_reusable_tokens)` across KV-indexer prefix matches. Warm requests match buckets where `extend_tokens = E` is within `[min_extend_tokens, max_extend_tokens]` and `input_tokens <= max_context_tokens` (falling back to `input_tokens` matching if no bucket admits `(E, L)`). Cold requests (`reusable == 0`) pool all workers across buckets whose `max_context_tokens` admits `L`.
- **Per-Rank Pending Work & `IdleSpill`**: `engine_reported_load.rs` tracks `min_rank_pending(w) = min_r (num_waiting_reqs(r) + num_waiting_uncached_tokens(r))`, which includes remaining uncached tokens of an in-flight chunked prefill. When every worker in `E`'s buckets has `min_rank_pending > 0`, `idle_spill_domains()` diverts the warm request (`decision="idle_spill"`) to context-compatible buckets with an idle DP rank (`min_rank_pending == 0`).

### 2.2 Mooncake `max_inflight_batches` Window and `_CompositeCompletionTicket`
In `mooncake-reshard/python/mooncake/reshard/transfer_engine/{executor.py,completion.py}` (`0977fc70`):
- **Bounded FIFO Overlap**: `TransferSubmission.execute_batches(..., max_inflight_batches=N)` submits batches via non-blocking `submit_scatter_read/write` and drains the oldest ticket once `len(published) >= N`. Across 8 batches (`2 ms` submit + `12 ms` transfer), `max_inflight_batches=4` reduces wall time from `112.0 ms` (`N=1`) to `34.0 ms` (`-69.64%`).
- **Shared Monotonic Deadline & GIL Quarantine**: `_CompositeCompletionTicket.drain(timeout_ms)` shares one `deadline_ns` across child tickets (`[1000, 600, 250] ms` remaining budgets for `[400, 350, 500] ms` children, capping total wait at `1000 ms`), while `_UndrainableCompletionUnknownTicket(owner)` keeps broken native tickets alive in memory so their C++ destructor never blocks under the GIL.

### 2.3 LMCache APC-Excluded Lookup Counters vs Request-Level Hit Rate
In `lmcache/v1/multiprocess/modules/lookup.py` (`168130df`), with `start_chunk = min(max(0, key.start // chunk_size), len(chunk_hashes))`:
- **LMCache Lookup Range (`[start_chunk, end)`)**: `apc_covered_tokens = start_chunk * chunk_size`, `requested_tokens = len(chunk_hashes) * chunk_size - apc_covered_tokens`, `l1_hit_tokens = min(l1_found_count, found_count) * chunk_size`, and `hit_tokens = (l1_chunks + l2_chunks) * chunk_size`.
- **Whole-Prompt Request View**: `request_hit_tokens = max(engine_hit_tokens, apc_covered_tokens + hit_tokens)` over `prompt_tokens` (taking `max` rather than summing because `engine_hit_tokens` and `apc_covered_tokens` both start at offset 0).

---

## 3. Architect Exercises & Falsifiable Questions

1. **Calculation**: Given `prompt_tokens = 2100`, `chunk_size = 256` (`8` chunks = `2048` tokens), `engine_computed_tokens = 600`, `key.start = 512` (`start_chunk = 2`), and `2` L1 + `3` L2 chunks found in `[2, 8)`, compute `requested_tokens`, `l1_hit_tokens`, `hit_tokens`, and `request_hit_tokens`.
   - **Answer**: `apc_covered_tokens = 512`, `requested_tokens = 1536`, `l1_hit_tokens = 512` (vs `1024` previously), `hit_tokens = 1280`, and `request_hit_tokens = max(600, 512 + 1280) = 1792` (`85.33%` of `2100`).
2. **Falsifiable Hypothesis**: Will `sgl-router` spill a warm request to a DP rank with `num_waiting_reqs = 0` and `num_waiting_uncached_tokens = 40,000`? No, because its pending prefill work is `40,000 > 0` (active chunked prefill remainder).
