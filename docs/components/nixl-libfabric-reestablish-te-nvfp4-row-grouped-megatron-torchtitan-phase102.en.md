# NIXL Libfabric Re-Establishment Handshake & Context-Parent Spans, TransformerEngine Graph-Safe Row-Scaled NVFP4 Grouped Cast & Megatron-LM / TorchTitan / LMCache Hardening (Phase 102)

- **Question ID**: `nixl-libfabric-reestablish-te-nvfp4-row-grouped-megatron-torchtitan-phase102` (`priority: 111`, component: `interconnect-networking`)
- **Evidence Status**: `own-offline` (First-hand upstream commit inspection + deterministic Python 3.12 stdlib model `experiments/nixl_libfabric_reestablish_te_nvfp4_row_phase102/run.py`)
- **Chinese Companion**: [`nixl-libfabric-reestablish-te-nvfp4-row-grouped-megatron-torchtitan-phase102.md`](nixl-libfabric-reestablish-te-nvfp4-row-grouped-megatron-torchtitan-phase102.md)

## 1. Primary Upstream Sources (Zero-Overlap)

1. **NIXL `9426653281` (#2438)**: `fix(libfabric): reply to re-establishment handshakes from connected peers`
2. **NIXL `02cc776992` (#2517)**: `tracing: begin a sampled post's span as the span its context identifies`
3. **TransformerEngine `89668d9dee` (#3649)**: `[Common][PyTorch] Add NVFP4 row-scaled grouped cast with graph-safe path`
4. **Megatron-LM `3ddfe0fe07` (#8516)**: `fix(fsdp): preserve retention parameter weight decay`
5. **TorchTitan `2ffc9fd3f8` (#5625)**: `[optim] Handle device modules without is_initialized() in the first step sync`
6. **LMCache `a5c2364820` (#6083)**: `[Fix] Validate trace schemas by level`
7. **LMCache `3f7c2017b1` (#6042)**: `[Docs][RBLN] Document the Rebellions NPU backend`
8. **Dynamo `3b7af264e4` (#17013)**: `test(operator): cover DisaggregatedSet and multinode LWS upgrade fixtures`

---

## 2. Core Implementation Mechanics & Quantitative Analysis

### 2.1 NIXL Libfabric Re-Establishment Handshake & Destructor Ordering (`9426653281`)

When vLLM's `NixlConnector` unilaterally evicts a remote engine after `engine_ttl = 3600s` and reconnects, it sends a handshake and waits up to **60.0s** for a reply. Previously, a peer still holding the existing connection never replied, causing every reconnect attempt to fail with `NIXL_ERR_REMOTE_DISCONNECT`.
- **CM-Thread Reply Queue (`pending_replies_`)**: Because `handleHandshake` runs on `progress_thread_` inside a completion context, it enqueues a reply onto `cm_thread_`, which sends a zero-`conn_info` handshake back on rail 0. During teardown, `~nixlLibfabricEngine()` stops `cm_thread_` **before** `progress_thread_` so in-flight reply sends can drain completions cleanly.
- **Recovery Latency**: Re-establishment drops from **60.0s timeout (`0/25` pass)** to **~0.1s (`25/25` pass, 600.0x faster recovery)**.

### 2.2 NIXL Context-Parent Spans (`02cc776992`)

`NIXL_TRACE_CONTEXT_SPAN` starts each sampled `nixl::postXferReq` span as the span identified by `call_context`, linking sender backend phases, `nixl::xfer.complete`, and receiver `nixl::notif.received` under one parent (`0` orphan child spans).

### 2.3 TransformerEngine Graph-Safe Row-Scaled NVFP4 Grouped Cast (`89668d9dee`)

`nvte_group_quantize` dispatches `kNVTEGroupedRowScaledNVFP4` to device-driven graph-safe amax and cast kernels (**32.0x** launch reduction for 32 experts, `64 → 2` launches) while enforcing 5 guards: `ndim == 2`, `128`-aligned `last_dim`, `<= 64` experts, full split coverage (`sum(splits) == rows`), and uniform quantizer scaling (preventing **504 B** out-of-bounds buffer corruption when mixing tensor-scaled and row-scaled quantizers).

### 2.4 Megatron-LM MFSDP v2, TorchTitan First-Step Sync & LMCache Trace Level Validation (`3ddfe0fe07`, `2ffc9fd3f8`, `a5c2364820`)

- **Megatron-LM (`3ddfe0fe07`)**: Preserves `is_wide_residual_retention_parameter` across `fully_shard` so 1-D retention parameters retain weight decay.
- **TorchTitan (`2ffc9fd3f8`)**: Guards `device_module.is_initialized()` for backends without lazy initialization (`torch.mps`) while still avoiding CUDA primary context creation when running CPU models on GPU hosts.
- **LMCache (`a5c2364820`)**: Enforces `TRACE_SCHEMA_VERSION` only for `level == "storage"` traces so `events` traces remain valid across storage API revisions.
