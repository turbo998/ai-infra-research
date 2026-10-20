# Mooncake Store Namespace & Lock-Free Read LeaseEvictionIndex, LMCache vLLM-RBLN 6-D Blocks-First KV Layout & Dynamo Text-Worker KV Router Guard / NIXL Explicit-Time Phase Events

**中文版**: [mooncake-namespace-lease-eviction-lmcache-rbln-blocks-first-dynamo-nixl-trace.md](mooncake-namespace-lease-eviction-lmcache-rbln-blocks-first-dynamo-nixl-trace.md)

## 1. Architectural Context & Problem Statement

Across multi-tenant distributed KV stores (Mooncake Store), NPU KV cache connectors (LMCache + Rebellions RBLN), and disaggregated serving routers/tracers (NVIDIA Dynamo & NIXL), four architectural gaps were resolved in upstream commits on `2026-10-20`:
1. **Mooncake Store Read-Path Eviction Index Contention & Tenant Handle Overhead**: Updating an eviction index on every lease-extending read serializes concurrent readers on an index lock, while resolving a single default tenant through a `shared_mutex` map and reference-counted handle dirties shared cache lines on every lookup.
2. **LMCache vLLM-RBLN 6-D Blocks-First KV Layout & Doubled DMA Pairs**: vLLM-RBLN's default attention kernels (`rbln_custom_ops`) allocate per-layer KV caches as 6-D blocks-first HND tensors `[num_blocks, 2, num_kv_heads, 1, block_size, head_size]` (with singleton axis 3 `== 1`), whereas LMCache only registered K/V-first `NL_X_TWO_NB_NH_ONE_BS_HS = 15` (`[2, NB, NH, 1, BS, HS]`), which also required two separate `memcpy_v2v` pairs per `(block, layer)` because `layer[:, block]` is non-contiguous.
3. **Dynamo Text-Input Worker Discovery with KV Routing**: Workers using backend tokenization (`vllm --use-vllm-tokenizer`, `ModelInput::Text`) bypass Dynamo's frontend tokenizer; admitting them under `RouterMode::KV` leaves the KV router without token IDs for prefix hashing.
4. **NIXL Backend Phase Timestamps & Communication Semantics**: `TracerPhaseSink::recordPhase` previously opened a `CorrelationScope` and recorded every backend phase as a `Kind::Metadata` span at the call clock rather than emitting a point event at the plugin's `steady_clock` timestamp with `CommSend`/`CommRecv` kinds.

## 2. First-Hand Implementation Audit

### 2.1 `kvcache-ai/Mooncake` (`d855a6aed9c8d1343af008b2cd9b9f4257d62a9b`, `#4721`, `2026-10-20`)
- **`mooncake-store/include/metadata/{namespace.h,namespace_table.h,lease_eviction_index.h,eviction_index.h,namespace_policy.h,object_route.h}`**:
  - `NamespaceTable` caches `default_namespace_` in a `std::atomic<Namespace*>` fast path (`0` locks, `0` shared cache-line writes);
  - `LeaseEvictionIndex` (`kBucketWidth = 100ms`, `observes_reads = false`) skips read hooks entirely: a read that extends a lease leaves the key in its older bucket (so its filed rank is never later than its real rank), and `StillAt` catches up under the key's write lock during eviction cursor walks, refiling the key to its true deadline bucket.

### 2.2 `LMCache/LMCache` (`462006abb998d4e4cb4209b14eb3d2db74a931b7`, `#5748`, `2026-10-20`)
- **`csrc/engine_kv_format.h`, `lmcache/v1/gpu_connector/kv_format/detectors/vllm.py`, `lmcache/v1/platform/devices/rbln/{kv_layout.py,kv_ops.py}`**:
  - Registers `EngineKVFormat.NL_X_NB_TWO_NH_ONE_BS_HS = 19` (`[NB, 2, NH, 1, BS, HS]`, `is_layer_list = true, is_hnd = true`), detects it purely from shape (`ndim == 6 and shape[1] == 2 and shape[3] == 1`), and warns when `shape[0] == 2 and shape[1] == 2` (`num_blocks == 2` ambiguity);
  - `_hnd_block_pairs` pairs contiguous `layer[block]` (`[2, H, BS, D]`) with `staged[position, layer_idx]`, cutting `memcpy_v2v` view pairs by **`-50.0%`** ($B \times L = 512$ vs $2 \times B \times L = 1,024$ for $B=16, L=32$).

### 2.3 `ai-dynamo/dynamo` (`1b6fee46c88b0f7339ac19423150d0613c82fa38`, `#16486`) & `ai-dynamo/nixl` (`705e0fc4f76993e9bbc5d39ed372db23c0c79e39`, `#2460`)
- **Dynamo `lib/llm/src/discovery/watcher.rs`**: Enforces `card.model_input != ModelInput::Text || !router_mode.is_kv_routing()` using `effective_router_config(card.router_config.as_ref(), &self.router_config)` during `ModelWatcher` discovery;
- **NIXL `src/core/tracing/{trace.h,trace_sink.cpp}`**: Adds `TraceBackend::emitSpan` and `emitEvent`, routing backend phases via `emitEvent` at the plugin's `nixlTime::getUs()` timestamp with `NOTIF_SENT -> Kind::CommSend` and `NOTIF_RECEIVED -> Kind::CommRecv`.

## 3. Deterministic Verification (`own-offline`)

Run `python experiments\mooncake_namespace_eviction_lmcache_rbln_dynamo_nixl\run.py --check`:

| Subsystem | Metric | Before / Baseline | After (`Phase 37`) | Evidence Tier |
|---|---|---|---|---|
| Mooncake `LeaseEvictionIndex` | Read Index Locks (10k reads) / 260ms Walk | Locks on every read | **`0` read locks; evicts 2 expired keys, refiles `k_read_extended -> bucket 6`** | `implementation` + `own-offline` |
| LMCache vLLM-RBLN 6-D HND | Format Enum / `B=16, L=32` `memcpy_v2v` Pairs | Format `15` only (`1,024` pairs) | **Format `19` (`[NB,2,NH,1,BS,HS]`), `512` pairs (`-50.0%`)** | `implementation` + `own-offline` |
| Dynamo `ModelWatcher` | `ModelInput::Text` + `RouterMode::KV` Guard | Admitted (broken KV hash) | **Rejected at discovery (`False`); card `RoundRobin` override allowed (`True`)** | `implementation` + `own-offline` |
| NIXL `TracerPhaseSink` | Phase Timestamp & `NOTIF_SENT/RECEIVED` Kind | Call-clock `Metadata` span | **Plugin `steady_clock` `at_us` + `CommSend` / `CommRecv` event** | `implementation` + `own-offline` |
