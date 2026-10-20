# Mooncake LocalHotCache Detached-Fill Scan Race Fix & NVIDIA Dynamo In-Place KV Block Hashing and Fail-Closed Prefill Activation

Deterministic Python 3.12 standard-library model verifying:
1. **`kvcache-ai/Mooncake` `ac4d29bff2e8bb98c36799aaf19e6c73529f0088` (`#4334`)**: `LocalHotCache::ReleaseHotKey` and `hasActiveBlockForKeyLocked` short-circuit reordering (`block && block->ref_count > 0 && block->key_ == key`) to prevent reading caller-owned `key_` on detached fill blocks (`ref_count == 0`).
2. **`ai-dynamo/dynamo` `1246a4bbd153de6d927cb074881afcc20d40ac15` (`#15701`)**: `hash_complete_blocks` in-place borrowed token-slice hashing (`0` bytes allocated for pure-text blocks, eliminating $1{,}048{,}576\,\text{B}$ of token clones at $128\text{K}$ tokens, and allocating only $13 \times \text{block\_size}$ bytes on multimodal-overlapping windows).
3. **`ai-dynamo/dynamo` `7aaf17b62691a5dc1141f71da1d5bca72ce6667d` (`#16736`)**: `PrefillRouter::activation_snapshot` atomic target/lifecycle/binding resolution under `self.target.lock()` and end-to-end `ErrorClass::WorkerUnavailable` (`HTTP 503`) preservation across Rust/PyO3 chat processors while prefill routing is activating.

## Run
```bash
python experiments/mooncake_hotcache_race_dynamo_kvhash_prefill_gate/run.py
python experiments/mooncake_hotcache_race_dynamo_kvhash_prefill_gate/run.py --check
```
