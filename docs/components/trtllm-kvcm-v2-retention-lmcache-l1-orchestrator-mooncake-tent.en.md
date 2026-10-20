# TensorRT-LLM KVCM V2 Native Retention (`cold_tier_offload_min_priority=30`), LMCache Shared L1 `MemoryOrchestrator` State Machine, and Mooncake TENT `Notify` Structured Ack Contract

**Language / 语言**: [English](trtllm-kvcm-v2-retention-lmcache-l1-orchestrator-mooncake-tent.en.md) | [中文](trtllm-kvcm-v2-retention-lmcache-l1-orchestrator-mooncake-tent.md)

> **Evidence Tier**: `implementation` (first-hand upstream commit inspection) + `own-offline` (Python 3.12 standard-library deterministic verification via `experiments/trtllm_kvcm_v2_retention_lmcache_l1_tent_notify/run.py --check`). No live multi-node RDMA hardware benchmark is claimed.

---

## 1. Executive Summary & Architectural Findings

Across hierarchical KV cache management (GPU HBM $\to$ Host DRAM $\to$ Shared L1 / remote storage) and RDMA control-plane notification RPCs, three upstream commits on 2026-10-20 established strict admission, fencing, and error-propagation invariants:

1. **TensorRT-LLM `KVCacheManagerV2` Native Retention & Cold-Tier Offload Gate ([`NVIDIA/TensorRT-LLM@41913522c7`](https://github.com/NVIDIA/TensorRT-LLM/commit/41913522c7f9f085c022421fb6183bd7592e076f))**:
   - **Native Block-Level Retention in V2**: Integrates `KvCacheRetentionConfig` (`0..100` priority range, default `kPriorityDefault = 35`, per-range `(priority, duration_ms)` with deadline expiry via `refreshRetention`) into `KVCacheManagerV2` (`kvCache.cpp`, `storageManager.cpp`).
   - **Cold-Tier Offload Threshold (`cold_tier_offload_min_priority = 30`)**: Introduces `kDefaultColdTierOffloadMinPriority = 30` (deprecating `secondary_offload_min_priority`). Unreferenced GPU pages with `priority < 30` are dropped directly instead of being offloaded to secondary cold storage, cutting cold-tier write bytes by **`50.0%`** (`1,572,864 -> 786,432 B`) in a representative 6-page mixed-priority eviction batch.
   - **Queue-Safe Priority Mutation & Helix CP Scaling**: `StorageManager::updatePriority` pins the page via `keepAlive = page.sharedFromThis()`, detaches it from its current priority LRU queue (`excludeFromEviction`), updates `page.mPriority`, re-schedules it (`scheduleForEviction`), and emits `EventManager::addPriorityUpdated`. Under Helix Context Parallelism (`_has_cp_helix`), `_retention_config_for_request` scales rank-local token intervals by `_helix_cp_size` to align with CP super-block ordinals, while `prefetchReuse(..., isPrefetch=true)` passes `claimRetention=false` so background tier prefetching never overwrites active retention priorities or deadlines.

2. **LMCache Shared L1 `MemoryOrchestrator` RegionState & Idempotent Replay Window ([`LMCache/LMCache@258746ce77`](https://github.com/LMCache/LMCache/commit/258746ce7750b085a8cec89d371ac7eaa4adb94d))**:
   - **`RegionState` Authority** (`lmcache/v1/memory_orchestrator/state.py`): Acts as the single metadata authority for a shared L1 memory region with power-of-two `alignment_bytes`, monotonic bump allocation (`_round_up(payload_bytes, alignment)`), and object lifecycle transitions `WRITING -> VALID` (or `CONSUMED` upon abort, TTL expiry, or client re-registration).
   - **All-or-Nothing Batch Reservation & Client Incarnation Fencing**: `reserve_write` computes total aligned `needed_bytes` across all absent keys in a batch; if `_allocated_bytes + needed_bytes > capacity_bytes`, every absent entry receives `OUT_OF_SPACE` and zero bytes are allocated. Re-registering an existing `client_id` with a new `client_incarnation` automatically retires the old incarnation (`_retire(previous)`), moving its in-flight `WRITING` objects to `CONSUMED` (`2` zombie writes retired in deterministic simulation) and releasing its read leases.
   - **Anti-Split-Brain Binding & `_ReplyWindow` Idempotency**: `server.py` enforces `_GRPC_OPTIONS = (("grpc.so_reuseport", 0),)` and a `<region_id>.marker` file (`reset_required=True` on unclean restart). `OrchestratorServicer` caches the latest `1024` mutating replies per incarnation keyed by `request_id -> (request_sha256, reply_bytes)`, replaying identical retries (`REPLAYED_OK`) while rejecting mutated payloads that reuse a `request_id` with `ALREADY_EXISTS`.

3. **Mooncake TENT `ControlService::onNotify` Structured Error Propagation ([`kvcache-ai/Mooncake@5cf142e344`](https://github.com/kvcache-ai/Mooncake/commit/5cf142e3447435faa9a9f6c846aea592bcfdc263))**:
   - Previously, `ControlService::onNotify()` in `mooncake-transfer-engine/tent/src/runtime/control_plane.cpp` left `response` empty for every `Notify` RPC, causing `ControlClient::notify()` to return `Status::OK()` even when `notify_callback_` was unregistered (during startup/teardown), JSON parsing failed, or the callback returned non-zero/threw (`3/3` silent false-positive acks).
   - Now `onNotify()` writes `"notify callback not registered"`, `"malformed notification: <err>"`, or `"notify callback failed: <err>"` into `response`, and `ControlClient::notify()` maps the `kMalformedNotification` prefix to non-retryable `Status::InvalidArgument` and other non-empty responses to retryable `Status::InternalError` (`0` silent false-positive acks).

---

## 2. Deterministic Offline Verification Summary (`own-offline`)

Verified via `python experiments/trtllm_kvcm_v2_retention_lmcache_l1_tent_notify/run.py --check` (`experiments/trtllm_kvcm_v2_retention_lmcache_l1_tent_notify/results/summary.json`):

| Subsystem | Stress / Fault Scenario | Legacy Behavior | Verified Implementation | Quantitative Invariant (`own-offline`) |
|---|---|---|---|---|
| **TRT-LLM `KVCacheManagerV2`** | 6 evicted pages (3 with priority `80/50/35` $\ge 30$, 3 with `25/20/10` $< 30$) | Offloads all 6 pages (`1,572,864 B`) | Drops `< 30` pages, offloads `786,432 B`; scales Helix CP ranges by `4x` | **`-50.0%`** cold-tier write bytes; `prefetch_claims_retention = False` |
| **LMCache `MemoryOrchestrator`** | Client `incarnation=1` crash with 2 `WRITING` objects + oversized batch + retry | Zombie `WRITING` locks or partial batch allocation | `incarnation=2` retires `2` writes to `CONSUMED`; atomic batch `OUT_OF_SPACE` | Identical retry `REPLAYED_OK`; mutated payload `ALREADY_EXISTS` |
| **Mooncake TENT `Notify` RPC** | Unregistered callback / malformed JSON / callback `rc = -5` | Returns `""` $\to$ `Status::OK()` (`3` false acks) | Maps parse failure to `InvalidArgument` and callback error to `InternalError` | Silent false-positive acks reduced from **`3 -> 0`** |

---

## 3. Architect Design Exercise & Falsifiable Follow-Up

1. **Control-Plane Failure Exercise**: During disaggregated PD startup, a prefill worker sends a `Notify` RPC to a decode worker whose TENT RPC server is up but has not yet registered its `notify_callback_`. How does `Mooncake@5cf142e344` classify this failure, and should the caller retry?
   - **Answer**: `onNotify` returns `"notify callback not registered"`, which `ControlClient::notify` maps to `Status::InternalError` (unlike `kMalformedNotification` which maps to `Status::InvalidArgument`), signaling the caller to back off and retry once initialization finishes.
2. **Falsifiable Follow-Up**: Because `RegionState._allocated_bytes` is a monotonic bump pointer in `[4/N]`, under what sustained `CONSUMED` churn rate does a shared L1 region exhaust `capacity_bytes` before follow-up extent compaction/eviction lands?
