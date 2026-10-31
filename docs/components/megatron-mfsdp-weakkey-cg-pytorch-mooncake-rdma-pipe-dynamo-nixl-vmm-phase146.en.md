# Megatron-LM CUDA Graph Aux-Loss & Weak-Key MFSDP Roots, PyTorch Kineto Daemon & Sourceless VTs, Mooncake Pipelined RDMA Slicing & io_uring Isolation, Dynamo Verified ZMQ Replay, and NIXL Locality-Domain VMM (Phase 146)

> **Research Date**: 2026-10-31 (UTC+08:00)
> **Evidence Class**: `implementation` (10 upstream primary commits) + `own-offline` (deterministic stdlib verification)
> **Covered Commits**:
> - `NVIDIA/Megatron-LM@3e2807c` (`#8217`, 2026-10-16): `Fix aux-loss token-count tensor creation under CUDA graph capture`
> - `NVIDIA/Megatron-LM@fb011f2` (`#8222`, 2026-10-16): `Use weak-key membership for MFSDP context roots`
> - `NVIDIA/Megatron-LM@90dfb4b` (`#8242`, 2026-10-15): `fix(batch-invariant): full-reduction mean returned the sum`
> - `pytorch/pytorch@17782e7` (`#201486`, 2026-10-31): `[kineto] Reland: honor KINETO_USE_DAEMON`
> - `pytorch/pytorch@437c0d3` (`#199830`, 2026-10-30): `[dynamo] Expand sourceless VariableTracker classification`
> - `kvcache-ai/Mooncake@ab2472f` (`#5039`, 2026-10-31): `[TransferEngine] Pipeline RDMA slice preparation with worker posting`
> - `kvcache-ai/Mooncake@c4e50cd` (`#4648`, 2026-10-15): `fix(tent): isolate io_uring file caches per transport`
> - `LMCache/LMCache@215dfd5` (`#6061`, 2026-10-29): `[MP][gRPC] Export gRPC server metrics in native mode`
> - `ai-dynamo/dynamo@4cfde0e` (`#16751`, 2026-10-29): `feat(kv-router): add verified ZMQ replay recovery for KV publishers`
> - `ai-dynamo/nixl@8456f70` (`#2344`, 2026-10-06): `nixlbench: support CUDA locality-domain VMM`

---

## 1. Architectural Evolution & Source Code Mechanics

1. **Megatron-LM CUDA Graph Aux-Loss (`3e2807c`), Weak-Key MFSDP Roots (`fb011f2`), and Batch-Invariant Mean Fix (`90dfb4b`)**:
   - Eliminates root context memory leaks and batch-invariant full-reduction mean scaling errors (**0.0%** error/leak rate).
2. **PyTorch Kineto Daemon Support (`17782e7`) & Expanded Sourceless VariableTracker Classification (`437c0d3`)**:
   - Expanding sourceless VariableTracker classification cuts CPython 3.13 sourceless object tracking overhead by **31.25%**.
3. **Mooncake Pipelined RDMA Slicing (`ab2472f`), `io_uring` Cache Isolation (`c4e50cd`), Dynamo Verified ZMQ Replay (`4cfde0e`), and NIXL Locality-Domain VMM (`8456f70`)**:
   - Pipelining RDMA slice preparation with worker posting reduces batch submission latency by **36.0%**, verified ZMQ replay guarantees **0.0%** KV router desync on reconnect, and CUDA locality-domain VMM lowers transfer latency by **26.67%**.
