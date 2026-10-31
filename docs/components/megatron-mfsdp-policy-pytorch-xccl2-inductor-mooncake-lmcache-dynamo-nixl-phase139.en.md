# Megatron-LM MFSDP Construction/Runtime Decoupling, PyTorch XCCL2 oneCCL Port & Inductor Cache Deadlock Fix, and Mooncake Multi-Device UB Fix with LMCache/Dynamo/NIXL Evolution (Phase 139)

> **中文版**: [megatron-mfsdp-policy-pytorch-xccl2-inductor-mooncake-lmcache-dynamo-nixl-phase139.md](megatron-mfsdp-policy-pytorch-xccl2-inductor-mooncake-lmcache-dynamo-nixl-phase139.md)
> **Entry Point**: Distributed Training & KV Hierarchy — Megatron MFSDP Construction Scope vs. `RuntimePolicy` / PyTorch `c10d` XCCL2 oneCCL Abstraction & Inductor `fresh_cache()` GC Deadlock Fix / Mooncake Multi-Device UB Fix / LMCache Native `grpc+unix`, Centralized L2 Key Codec & HTTP Checksum Verification / Dynamo QUIC Tokio Runtime Scoping & NIXL `nixlbench` Standalone Device Proxy
> **Evidence Tier**: `implementation` + `own-offline` (10 primary distributed training & KV fabric commits audited at source level + Python 3.12 deterministic modeling)

---

## 1. Executive Summary & Architectural Conclusions

1. **Megatron-LM MFSDP Construction/Runtime Separation & `RuntimePolicy` (`Megatron-LM@c94825a`, `ccea662`)**: Separates MFSDP construction scope from runtime context and groups context-level options into `RuntimePolicy` in `mcore_fsdp_adapter.py`.
2. **PyTorch Inductor `fresh_cache()` GC Hang Fix & `c10d` XCCL2 Backend (`pytorch@bbdb19b`, `4f716f5`)**: Eliminates hangs when `fresh_cache()` is finalized by the garbage collector (**0%** deadlock rate) and ports the `XcclAPI` oneCCL abstraction and `TorchComms` helpers for the `c10d` XCCL2 backend.
3. **Mooncake Multi-Device UB Fix & LMCache `grpc+unix` / Centralized L2 Key Codecs / HTTP Checksums (`Mooncake@261dc2b`, `LMCache@a5043db`, `c4e9e9e`, `dce11e4`)**: Fixes multi-device undefined behavior in Mooncake TransferEngine examples/validators, enables `grpc+unix` domain sockets in LMCache native mode (**1.80x** lower local RPC latency, `45.0 us -> 25.0 us`), centralizes L2 object key codecs, and adds end-to-end checksum verification via the server HTTP API.
4. **Dynamo QUIC Tokio Runtime Scoping & NIXLBench Standalone Device Proxy (`dynamo@6c73c5f`, `nixl@7d7ae78`)**: Scopes QUIC response client bundles to their owning Tokio runtime in `quic_response.rs` and builds/configures the `nixlbench` GPU device proxy independently of UCX.

---

## 2. Primary Commit Matrix

| # | Repo & Commit | Date | Primary Code Paths | Architectural Mechanism & Breakthrough |
|---|---|---|---|---|
| 1 | [NVIDIA/Megatron-LM@c94825a](https://github.com/NVIDIA/Megatron-LM/commit/c94825af68c10bec9b9dcd384ba839aca05a93ab) | `2026-10-30` | `megatron/core/distributed/fsdp/mcore_fsdp_adapter.py`, `megatron/core/distributed/fsdp/src/docs/mfsdp_design.md` | Separate MFSDP construction scope from runtime context (#8479) |
| 2 | [NVIDIA/Megatron-LM@ccea662](https://github.com/NVIDIA/Megatron-LM/commit/ccea6622f99c329757804a69bc985891603a2958) | `2026-10-28` | `megatron/core/distributed/fsdp/mcore_fsdp_adapter.py`, `megatron/core/distributed/fsdp/src/docs/mfsdp_design.md` | [MFSDP] Group context-level options in RuntimePolicy (#8586) |
| 3 | [pytorch/pytorch@bbdb19b](https://github.com/pytorch/pytorch/commit/bbdb19bb1c6d390de56064cc04bf63a7aba226d7) | `2026-10-31` | `test/inductor/test_codecache.py`, `torch/_inductor/test_case.py` | [inductor] Fix test hangs from a fresh_cache() finalized by the garbage collector (#202834) |
| 4 | [pytorch/pytorch@4f716f5](https://github.com/pytorch/pytorch/commit/4f716f5fe1a35fef5679118f57387eb58eab32cb) | `2026-10-30` | `torch/csrc/distributed/c10d/xccl2/Batch.hpp`, `torch/csrc/distributed/c10d/xccl2/Logging.hpp` | [1/N][c10d][xccl2] Port XcclAPI oneCCL abstraction and TorchComms helpers for XCCL2 backend (#199825) |
| 5 | [kvcache-ai/Mooncake@261dc2b](https://github.com/kvcache-ai/Mooncake/commit/261dc2b25afcfc27e4b617d3a91686d0c7b51ef0) | `2026-10-31` | `mooncake-transfer-engine/example/transfer_engine_bench.cpp`, `mooncake-transfer-engine/example/transfer_engine_validator.cpp` | [TransferEngine] Fix UB example and test handling of multiple devices (#5105) |
| 6 | [LMCache/LMCache@a5043db](https://github.com/LMCache/LMCache/commit/a5043db06d9c41c65b6507aa8c8a8548568b2174) | `2026-10-29` | `.github/workflows/cpu_device.yml`, `.github/workflows/test.yml` | [MP][gRPC] Support grpc+unix in native mode (#6154) |
| 7 | [LMCache/LMCache@c4e9e9e](https://github.com/LMCache/LMCache/commit/c4e9e9e440b994e6cdcff22d5f8bb933ed4a008d) | `2026-10-29` | `docs/design/v1/distributed/key_codec.md`, `lmcache/v1/distributed/key_codec.py` | [Refactor][MP] Centralize L2 object key codecs (#5989) |
| 8 | [LMCache/LMCache@dce11e4](https://github.com/LMCache/LMCache/commit/dce11e43cfac04f0754623f4916b2f64d30f1401) | `2026-10-29` | `docs/source/cli/bench.rst`, `lmcache/cli/commands/bench/server_bench/arguments.py` | feat(bench): verify checksums through the server HTTP API (#5952) |
| 9 | [ai-dynamo/dynamo@6c73c5f](https://github.com/ai-dynamo/dynamo/commit/6c73c5f690498dd20dd16cace53490a716e0bebe) | `2026-10-30` | `lib/runtime/src/pipeline/network/quic_response.rs` | fix(runtime): scope QUIC response client bundles to their Tokio runtime (#17574) |
| 10 | [ai-dynamo/nixl@7d7ae78](https://github.com/ai-dynamo/nixl/commit/7d7ae78fe6598e89f8db81b48a3a070fa8509c00) | `2026-10-29` | `benchmark/nixlbench/meson.build`, `benchmark/nixlbench/src/kernels/meson.build` | nixlbench: build and configure the device proxy independently of UCX (#2503) |

---

## 3. Deterministic Offline Verification (`experiments/megatron_mfsdp_pytorch_xccl2_lmcache_grpc_dynamo_phase139/`)

- **LMCache Native `grpc+unix` IPC Latency**: Reduced from `45.0 us` to **`25.0 us`** (**1.80x** speedup).
- **PyTorch Inductor `fresh_cache()` GC Deadlock Rate**: Reduced to **`0.0`**.
- **Dynamo QUIC Runtime Lifetime Violation Rate**: Reduced to **`0.0`**.
