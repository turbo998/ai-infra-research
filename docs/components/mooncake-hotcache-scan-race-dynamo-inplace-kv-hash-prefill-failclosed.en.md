# Deep Dive: Mooncake LocalHotCache Detached-Fill Scan Race Fix, NVIDIA Dynamo In-Place Token-Block KV Hashing, and Fail-Closed Prefill Activation

> **中文版**: [mooncake-hotcache-scan-race-dynamo-inplace-kv-hash-prefill-failclosed.md](mooncake-hotcache-scan-race-dynamo-inplace-kv-hash-prefill-failclosed.md) | **Back to Index**: [README.en.md](../../README.en.md) | **7-Card Benchmark Report**: [ai-infra-phase33-benchmark-cards.html](../reports/ai-infra-phase33-benchmark-cards.html)

## 1. First-Hand Source Matrix (2026-10-20)

| Component | Repository & Commit SHA | Release Date | Key Files & Symbols | Evidence Class |
| :--- | :--- | :--- | :--- | :--- |
| **Mooncake Store Detached Hot-Cache Fill Key-Scan Race Fix** | `kvcache-ai/Mooncake` [`ac4d29bff2e8bb98c36799aaf19e6c73529f0088`](https://github.com/kvcache-ai/Mooncake/commit/ac4d29bff2e8bb98c36799aaf19e6c73529f0088) (`#4334`) | `2026-10-20` | `mooncake-store/src/local_hot_cache.cpp::LocalHotCache::ReleaseHotKey`, `hasActiveBlockForKeyLocked` | `implementation` + `own-offline` |
| **NVIDIA Dynamo In-Place Token-Block KV Hashing (`hash_complete_blocks`)** | `ai-dynamo/dynamo` [`1246a4bbd153de6d927cb074881afcc20d40ac15`](https://github.com/ai-dynamo/dynamo/commit/1246a4bbd153de6d927cb074881afcc20d40ac15) (`#15701`) | `2026-10-20` | `lib/tokens/src/lib.rs::hash_complete_blocks`, `lib/kv-hashing/src/compute.rs::Request::block_lineages` | `implementation` + `own-offline` |
| **NVIDIA Dynamo PrefillRouter Fail-Closed Activation & `WorkerUnavailable` (HTTP 503)** | `ai-dynamo/dynamo` [`7aaf17b62691a5dc1141f71da1d5bca72ce6667d`](https://github.com/ai-dynamo/dynamo/commit/7aaf17b62691a5dc1141f71da1d5bca72ce6667d) (`#16736`) | `2026-10-20` | `lib/llm/src/kv_router/prefill_router/activation.rs::ActivationSnapshot`, `lib/bindings/python/rust/errors.rs::routed_error_to_pyerr` | `implementation` + `own-offline` |

---

## 2. Mooncake `LocalHotCache` Detached-Fill Key-Scan Data Race Prevention

In Mooncake Store's `LocalHotCache`, `GetFreeBlock()` detaches a free `HotMemBlock` (`ref_count == 0`) from the tail so a `SubmitPutTask` worker thread can populate its payload and mutate `filling->key_` **without holding `mutex_`** before publishing it via `PutHotKey`.

Previously, `ReleaseHotKey` and `hasActiveBlockForKeyLocked` scanned `blocks_` under `mutex_` using:
```cpp
if (block && block->key_ == key && block->ref_count > 0)
```
Because `block->key_ == key` executed before `block->ref_count > 0`, any concurrent scan read `block->key_` on detached fill blocks while the filling thread was reallocating that `std::string` without a lock—a C++ data race (`local_hot_cache_concurrency_test.cpp`).

Commit `ac4d29bff2e8` (`#4334`) reorders the short-circuit check to `if (block && block->ref_count > 0 && block->key_ == key)`. Since detached fill blocks have `ref_count == 0`, their caller-owned `key_` is never touched during key scans (reducing racy unlocked key reads from `16,000` to `0` across 1,000 scans in our 64-block model).

---

## 3. NVIDIA Dynamo In-Place Token-Block KV Hashing (`hash_complete_blocks`)

Previously, `Request::into_blocks`, `block_hashes`, and `sequence_hashes` built a full `TokenBlockSequence::new_with_mm(Tokens::from(self.tokens.clone()), ...)`, cloning the entire request token vector and allocating a separate `Vec<Token>` inside every `TokenBlock` only to discard the copied tokens after reading their hashes.

Commit `1246a4bbd153` (`#15701`) introduces `hash_complete_blocks(&self.tokens, &token_mm, block_size, salt_hash)` in `lib/tokens/src/lib.rs`:
- Advances a single monotonic `run_cursor` across sorted, non-overlapping `mm_runs`.
- **Pure-text windows**: hashed in place directly from the borrowed slice `&tokens[block_offset..block_end]` via `compute_block_hash_for_tokens` (**`0` bytes cloned or allocated**).
- **Multimodal-overlapping windows**: allocate only the $13 \times \text{block\_size}$ byte tagged frame (`compute_block_bytes_with_mm`).
- At $128\text{K}$ tokens (`block_size = 128`, 1,024 blocks), token heap allocations drop from **$1{,}048{,}576\,\text{B}$ (1,025 allocations) to $0\,\text{B}$ (0 allocations)**; on an $8\text{K}$ multimodal request with a 32-token image span (`mm_mid_8k_block_16`), allocations drop by **$99.37\%$** ($65{,}952\,\text{B} \to 416\,\text{B}$ across only the 2 overlapping blocks).

---

## 4. NVIDIA Dynamo PrefillRouter Fail-Closed Activation & HTTP 503 Preservation

During disaggregated rollout or startup, decode workers can become routable shortly before the frontend finishes constructing the prefill router binding. Previously, lock-free checks treated an activating prefill router as inactive and passed requests straight to disaggregated-only decode workers, which rejected them.

Commit `7aaf17b62691` (`#16736`) introduces:
1. **`PrefillRouter::activation_snapshot()`**: Reads `(target, lifecycle, binding)` atomically under `self.target.lock()`, returning `ActivationSnapshot::{Active(Arc<PrefillBinding>), Pending, Untargeted}`. When a prefill target is set (`Pending`) but its binding is not yet published, the router fails closed with `ErrorClass::WorkerUnavailable`.
2. **End-to-End PyO3 HTTP 503 Preservation**: Because `DynamoError::class()` normalizes `WorkerUnavailable` to `Unavailable`, `is_worker_unavailable()` inspects `error.error_type() == ErrorClass::WorkerUnavailable` across the cause chain, converts it via `routed_error_to_pyerr` into `dynamo._core.WorkerUnavailable`, re-raises it unchanged through `vllm_processor.py` and `sglang_processor.py`, and restores `ErrorClass::WorkerUnavailable` in `worker_unavailable_to_dynamo` so clients receive a retryable **HTTP 503** instead of an HTTP 500 envelope.

---

## 5. Reproducible Experiment

```bash
python experiments/mooncake_hotcache_race_dynamo_kvhash_prefill_gate/run.py --check
```
