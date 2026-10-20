# Experiment: TensorRT-LLM KVCM V2 Native Retention, LMCache Shared L1 `MemoryOrchestrator` & Mooncake TENT `Notify` Ack Contract (`trtllm_kvcm_v2_retention_lmcache_l1_tent_notify`)

Deterministic Python 3.12 standard-library verification for Phase 34 (`priority: 43`) Topic 3:
- **`NVIDIA/TensorRT-LLM@41913522c7`**: `KVCacheManagerV2` native per-block `KvCacheRetentionConfig` priority (`0..100`, default `35`), `cold_tier_offload_min_priority=30` gate dropping low-priority pages (`-50.0%` cold-tier write bytes in mixed priority workloads), Helix CP token-range scaling (`token * _helix_cp_size`), and `isPrefetch=true` non-claiming reuse.
- **`LMCache/LMCache@258746ce77`**: `RegionState` monotonic power-of-two aligned extent allocation, `WRITING -> VALID -> CONSUMED` lifecycle, `client_incarnation` retirement of zombie writes/leases, all-or-nothing batch `OUT_OF_SPACE`, and `_ReplyWindow` (`SHA-256` request digest) idempotent RPC replay (`REPLAYED_OK` vs `ALREADY_EXISTS`).
- **`kvcache-ai/Mooncake@5cf142e344`**: `ControlService::onNotify` & `ControlClient::notify` structured failure propagation (`InvalidArgument` on `kMalformedNotification`, `InternalError` on unregistered or failed callback), eliminating `3/3` silent false-positive `Status::OK()` delivery acks.

## Run
```bash
python experiments/trtllm_kvcm_v2_retention_lmcache_l1_tent_notify/run.py --check
```
