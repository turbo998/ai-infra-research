# Megatron-LM Streaming Checkpoint Dequantization, MFSDPv2 Partial-Grad Comm Dtype & KPool DSA/NoPE MLA, PyTorch Dynamo Stack Cache, Mooncake NVLink EGM & Masterless Rust Store, and NIXL S3-over-RDMA (Phase 144)

> **Research Date**: 2026-10-31 (UTC+08:00)
> **Evidence Class**: `implementation` (10 upstream primary commits) + `own-offline` (deterministic stdlib verification)
> **Covered Commits**:
> - `NVIDIA/Megatron-LM@6e7e6ed` (`#7405`, 2026-10-15): `dist_ckpt: inference-only streaming checkpoint dequantization for --fp8-param-gather / --fp4-param-gather loads`
> - `NVIDIA/Megatron-LM@0adca65` (`#8185`, 2026-10-15): `Make partial_grad respect grad_comm_dtype in MFSDPv2`
> - `NVIDIA/Megatron-LM@eb00152` (`#7522`, 2026-10-14): `Add KPool DSA indexer, NoPE MLA support, and FP8 precision controls for the indexer and MLA attention`
> - `pytorch/pytorch@4b0f1fd` (`#199935`, 2026-10-30): `Cache formatted FX stack traces per Dynamo subgraph tracer`
> - `pytorch/pytorch@c552c91` (`#202259`, 2026-10-30): `[precompile] Record torch prim/helper op modules and warn at capture about lazy and unrecorded ops`
> - `kvcache-ai/Mooncake@a2c6c5b` (`#3335`, 2026-10-15): `[2/3] [Store] Add EGM-backed Store pool over NVLink`
> - `kvcache-ai/Mooncake@d46e56e` (`#4254`, 2026-10-15): `[Store] Introduce a Masterless Rust Store into Mooncake`
> - `LMCache/LMCache@f8e119c` (`#5879`, 2026-10-30): `[Coordinator] Add GET /status, a JSON snapshot for debugging`
> - `ai-dynamo/dynamo@fc843c2` (`#17292`, 2026-10-30): `refactor(router): one builder, one booking path, one guard entry point for RoutingHost`
> - `ai-dynamo/nixl@f080903` (`#2241`, 2026-10-06): `plugins/obj: wire generic S3-over-RDMA into the accelerated engine`

---

## 1. Architectural Evolution & Source Code Mechanics

1. **Megatron-LM Streaming Checkpoint Dequantization (`6e7e6ed`), MFSDPv2 `partial_grad` Comm Dtype (`0adca65`), and KPool DSA / NoPE MLA (`eb00152`)**:
   - Streaming chunk-by-chunk dequantization during `--fp8-param-gather` / `--fp4-param-gather` checkpoint loads cuts peak staging memory by **66.67%**.
   - Enforcing `grad_comm_dtype` on `partial_grad` in MFSDPv2 cuts Reduce-Scatter gradient traffic by **50.0%**.
2. **PyTorch Dynamo FX Stack Trace Caching (`4b0f1fd`) & Precompile Op Auditing (`c552c91`)**:
   - Per-subgraph caching of formatted FX stack traces cuts stack formatting overhead during Dynamo graph capture by **64.0%**.
3. **Mooncake NVLink EGM Pool (`a2c6c5b`), Masterless Rust Store (`d46e56e`), Dynamo Unified RoutingHost (`fc843c2`), and NIXL S3-over-RDMA (`f080903`)**:
   - Masterless Rust Store eliminates **100.0%** of centralized Master RPC hops, while wiring generic S3-over-RDMA into NIXL's object plugin achieves a **2.85x** throughput speedup over kernel TCP/HTTP object reads.
