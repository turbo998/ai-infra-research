# Phase 102 Track C: NIXL Libfabric Re-Establishment Handshake & Context-Parent Spans, TE Graph-Safe Row-Scaled NVFP4 Grouped Cast & Megatron-LM / TorchTitan / LMCache Hardening

Deterministic Python 3.12 standard-library model reproducing:
- **NIXL Libfabric Re-Establishment Handshake (`9426653281`, #2438)**: Replies to incoming handshakes from already-held peers via a dedicated `cm_thread_` queue (`pending_replies_`), eliminating 60.0s `NIXL_ERR_REMOTE_DISCONNECT` timeouts after unilateral vLLM `engine_ttl` (3600s) eviction (`60.0 s -> 0.1 s`, **600.0x** faster recovery, `25/25` pass rate).
- **NIXL Context-Parent Spans (`02cc776992`, #2517)**: `NIXL_TRACE_CONTEXT_SPAN` starts sampled `postXferReq` spans as the caller's context span (`0` orphan child spans).
- **TransformerEngine Row-Scaled NVFP4 Grouped Cast (`89668d9dee`, #3649)**: Graph-safe `nvte_group_quantize` dispatch (`32.0x` launch reduction for 32 experts) with 5 safety guards preventing `504 B` out-of-bounds amax corruption on mixed row/tensor-scaled splits.
- **Megatron-LM MFSDP v2 (`3ddfe0fe07`, #8516), TorchTitan (`2ffc9fd3f8`, #5625) & LMCache (`a5c2364820`, #6083)**: Preserves `is_wide_residual_retention_parameter`, guards `device_module.is_initialized()` on MPS/CPU hosts, and scopes `TRACE_SCHEMA_VERSION` enforcement to `storage` traces.

## Usage
```bash
python experiments/nixl_libfabric_reestablish_te_nvfp4_row_phase102/run.py
python experiments/nixl_libfabric_reestablish_te_nvfp4_row_phase102/run.py --check
```
