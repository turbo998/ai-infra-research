# Mooncake RDMA Data-Path Auto-GID Epoch Recovery, LMCache In-Process GPU Connector Isolation, and NIXL UCX Closed-Vocabulary Phase Tracing

> **中文版**: [mooncake-auto-gid-recovery-reshard-inflight-lmcache-nixl-ucx-spans.md](mooncake-auto-gid-recovery-reshard-inflight-lmcache-nixl-ucx-spans.md)  
> **Evidence Class**: `implementation` (first-hand upstream commits) + `own-offline` (`experiments/mooncake_gid_recovery_lmcache_inproc_nixl_ucx/run.py --check`)

## 1. First-Hand Source Code & Root-Cause Mechanics

1. **Mooncake TransferEngine Epoch-Stamped Auto-GID Recovery on `IBV_WC_RETRY_EXC_ERR`** (`kvcache-ai/Mooncake` commit [`5ea6cf0d95d3a69b4ba196d28bf1f96e28ec5e10`](https://github.com/kvcache-ai/Mooncake/commit/5ea6cf0d95d3a69b4ba196d28bf1f96e28ec5e10), `2026-10-20`):
   - **Code Path**: `mooncake-transfer-engine/include/transport/rdma_transport/rdma_gid_probe.h`, `worker_pool.cpp` (`performPostSend`, `processCompletions`, `recordAutoGidDataPathFailure`), and `transport.h` (`RdmaCompletionToken`).
   - **Mechanism**: In multi-homed or dual-stack RoCEv2 clusters, an initially probed GID may complete handshake metadata exchange yet fail on data-path RDMA writes with `IBV_WC_RETRY_EXC_ERR`. Previously, `WorkerPool` immediately called `markRailFailed` and blacklisted the NIC rail. Commit `5ea6cf0d` stamps each posted `Slice` with `RdmaCompletionToken{endpoint_id, selection_epoch, valid, observe_success}`. On `IBV_WC_RETRY_EXC_ERR`, the first completion matching the active `selection_epoch` advances the endpoint to the next `AutoGidCandidateClass` candidate (`kAdvanceGid`), while subsequent failed slices from the same in-flight burst carry the stale `selection_epoch` and return `kIgnoreDuplicate` so they do not exhaust the remaining candidate GIDs.

2. **LMCache In-Process GPU Connector Isolation & Upfront Device Feature Validation** (`LMCache/LMCache` commit [`df19cd68cd959510b36de88c72b526232359505d`](https://github.com/LMCache/LMCache/commit/df19cd68cd959510b36de88c72b526232359505d), `2026-10-20`):
   - **Code Path**: `lmcache/inprocess/v1/gpu_connector/__init__.py` (`_validate_vllm_device_features`), `lmcache/v1/gpu_connector/__init__.py`, and `lmcache/integration/vllm/vllm_service_factory.py`.
   - **Mechanism**: Commit `df19cd68` isolates vLLM-coupled in-process GPU connectors (`VLLMPagedMemGPUConnectorV2/V3`, MUSA, XPU, HPU) into `lmcache/inprocess/v1/gpu_connector/` so standalone `lmcache/v1/gpu_connector` imports cleanly on CPU-only or multiprocess builds. Furthermore, `_validate_vllm_device_features` rejects unsupported combinations (`enable_blending` or `use_gpu_connector_v3` outside `{cuda, xpu}`, and `use_layerwise=True` on `hpu`) with an explicit `ValueError` before any `torch.cuda.Stream()` crash can occur.

3. **NIXL UCX Closed-Vocabulary Sender/Receiver Phase Tracing Without Poll-Skew** (`ai-dynamo/nixl` commit [`8d5660c118cf2d5249edd83d9afc311cdda2e43d`](https://github.com/ai-dynamo/nixl/commit/8d5660c118cf2d5249edd83d9afc311cdda2e43d), `2026-10-20`):
   - **Code Path**: `src/plugins/ucx/ucx_backend.cpp` (`sampledTraceContext`, `recordPhase`, `postXfer`), `src/plugins/ucx/ucx_backend_req.h`, and `docs/tracing.md`.
   - **Mechanism**: Under sampled `TraceContext`, UCX emits `nixl::submit` when `postXfer` accepts a request, `nixl::wire.submitted` after `sendXferRange` posts all puts/gets and flushes (omitted if peer disconnect fails submission), and `nixl::notif.sent` once per notifying transfer, paired with receiver `nixl::notif.received`. UCX deliberately omits `nixl::wire.completed` because completion is only observed when the application polls status, which would time the poll loop rather than the transport wire duration.

## 2. Deterministic Verification (`experiments/mooncake_gid_recovery_lmcache_inproc_nixl_ucx`)

| Sub-Track | Metric | Legacy | Patched | Architectural Impact |
|---|---|---:|---:|---|
| Mooncake Auto-GID Recovery (`5ea6cf0d`) | 16-slice burst recovery on `IBV_WC_RETRY_EXC_ERR` | `0 / 16` (NIC rail blacklisted) | `16 / 16` (`epoch 1->2`, `15` stale dups ignored) | Recovers healthy RoCEv2 GID on the same NIC rail without false rail eviction |
| LMCache In-Process Isolation (`df19cd68`) | Invalid accelerator configs caught upfront / Deep stream crashes | `0 / 3` caught / `3` crashes | `3 / 3` `ValueError` / `0` crashes | Clean separation of standalone vs in-process connectors across CUDA/XPU/MUSA/HPU |
| NIXL UCX Phase Tracing (`8d5660c1`) | Application poll-delay inflation over true wire time | `195.5 us` vs `23.5 us` (**`8.3191x` skew**) | Emits `submit` / `wire.submitted` / `notif.sent` only | Eliminates status-polling skew from RDMA transport telemetry |
