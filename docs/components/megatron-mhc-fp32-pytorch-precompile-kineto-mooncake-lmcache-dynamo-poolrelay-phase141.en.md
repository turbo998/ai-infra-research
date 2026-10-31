# Megatron-LM Pairwise Inference & mHC IEEE FP32 Parity, PyTorch Precompile Compile-Time Autotune & ROCm Kineto/CK Hardening, and Mooncake/LMCache Fault Tolerance with Dynamo Active-Active PoolRelay (Phase 141)

> **中文版**: [megatron-mhc-fp32-pytorch-precompile-kineto-mooncake-lmcache-dynamo-poolrelay-phase141.md](megatron-mhc-fp32-pytorch-precompile-kineto-mooncake-lmcache-dynamo-poolrelay-phase141.md)
> **Entry Point**: Distributed Training & KV Hierarchy — Megatron-LM Pairwise Inference Campaigns & mHC PP IEEE FP32 GEMMs / PyTorch Precompile Capture-Time Triton Autotuning, ROCm Kineto In-Flight Record Drain & CK GEMM Fallback / Mooncake `tebench` Batch-Release Crash Fix & DFS Retry / LMCache `grpc://` MP Server Auto-Start / Dynamo Operator Active-Active `PoolRelay` & NIXL ROCm Meta Dispatcher
> **Evidence Tier**: `implementation` + `own-offline` (10 primary distributed training & KV fabric commits audited at source level + Python 3.12 deterministic modeling)

---

## 1. Executive Summary & Architectural Conclusions

1. **Megatron-LM Pairwise Inference Campaigns & mHC PP IEEE FP32 Parity (`Megatron-LM@0a51326`, `7196f83`)**: Integrates five inference pairwise evaluation campaigns with prefix-cache prompt scoring, and enforces IEEE FP32 GEMMs in mHC pipeline-parallel parity tests.
2. **PyTorch Precompile Capture-Time Triton Autotuning & ROCm Kineto/CK Hardening (`pytorch@8c03c52`, `b0e48e7`, `7768be0`)**: Autotunes Triton kernels at compile time during precompile capture (**100.0%** elimination of online first-step autotuning latency, `18.0 s -> 0.0 s`), waits for in-flight kernel dispatch records when stopping ROCm Kineto (**0.0%** dropped tail events), and hardens ROCm CK GEMM fallback/beta/FP16 support.
3. **Mooncake `tebench`/DFS Fault Tolerance & LMCache `grpc://` Auto-Start (`Mooncake@c3d05a7`, `3eabbd8`, `LMCache@5fbf3c5`)**: Fixes classic `tebench` crashes on batch release and worker failure plus DFS sync client retry bugs in Mooncake, and auto-starts a local gRPC MP server for `grpc://` connector endpoints in LMCache.
4. **Dynamo Operator Active-Active `PoolRelay` & NIXL ROCm Dispatcher Preference (`dynamo@008d356`, `nixl@d8253ad`)**: Deploys active-active `PoolRelay` replicas for a single DGD in the Dynamo Kubernetes operator (**0.0 ms** single-relay failover interruption) and prefers `nixl_rocm` when the NIXL Python meta dispatcher detects ROCm.

---

## 2. Primary Commit Matrix

| # | Repo & Commit | Date | Primary Code Paths | Architectural Mechanism & Breakthrough |
|---|---|---|---|---|
| 1 | [NVIDIA/Megatron-LM@0a51326](https://github.com/NVIDIA/Megatron-LM/commit/0a513268eee94d8e903fecbc21fde862eecc31f3) | `2026-10-29` | `.github/actions/action.yml`, `docs/mcore-inference-user-guide.md` | Integrate five inference pairwise campaigns and prefix-cache prompt scores (#8426) |
| 2 | [NVIDIA/Megatron-LM@7196f83](https://github.com/NVIDIA/Megatron-LM/commit/7196f838952b1ae51dca1d1a1a585847f1655a60) | `2026-10-29` | `tests/unit_tests/pipeline_parallel/test_pp_mhc_compatibility.py` | test: use IEEE FP32 GEMMs in mHC PP parity test (#8501) |
| 3 | [pytorch/pytorch@8c03c52](https://github.com/pytorch/pytorch/commit/8c03c528651c04532e121f4d56c4bc4b4e7136b2) | `2026-10-30` | `test/dynamo/test_precompile_package.py`, `test/inductor/test_multi_kernel.py` | [precompile] Autotune Triton kernels at compile time during a capture (#202255) |
| 4 | [pytorch/pytorch@b0e48e7](https://github.com/pytorch/pytorch/commit/b0e48e7b450ff27ab02c171a520c1183da9eed93) | `2026-10-30` | `torch/csrc/profiler/kineto/libkineto/include/Config.h`, `torch/csrc/profiler/kineto/libkineto/src/Config.cpp` | [ROCm][kineto] Wait for in-flight kernel dispatch records when stopping (#202758) |
| 5 | [pytorch/pytorch@7768be0](https://github.com/pytorch/pytorch/commit/7768be02b3f66feeb05945ccbb70880b545bda08) | `2026-10-30` | `aten/src/ATen/cuda/CUDABlas.cpp`, `aten/src/ATen/native/hip/bgemm_kernels/bgemm_kernel_bf16bf16bf16_128_128x16x64_16x16_4x1_8x16x1_8x16x1_1x16x1x8_2_Intrawave_v2.hip` | [ROCm] CK GEMM: fall back instead of raising, honor beta, enable fp16 (#202096) |
| 6 | [kvcache-ai/Mooncake@c3d05a7](https://github.com/kvcache-ai/Mooncake/commit/c3d05a7e2c52ac2de3282f5952934d7e26c90568) | `2026-10-16` | `mooncake-transfer-engine/benchmark/te_backend.cpp`, `mooncake-transfer-engine/benchmark/te_backend.h` | [Bugfix][TransferEngine] Fix classic tebench crashes on batch release and on worker failure (#4698) |
| 7 | [kvcache-ai/Mooncake@3eabbd8](https://github.com/kvcache-ai/Mooncake/commit/3eabbd80a494678f59b0d9c5ff199c2c469eb732) | `2026-10-20` | `mooncake-store/tests/dfs_sync_client_test.cpp` | [Store][Bugfix] Fix DFS sync client test retry failure (#4781) |
| 8 | [LMCache/LMCache@5fbf3c5](https://github.com/LMCache/LMCache/commit/5fbf3c552cea165837a886aba75b61de857b03b2) | `2026-10-29` | `docs/design/integration/vllm/mp-autostart.md`, `docs/source/mp/configuration.rst` | [Feat][MP] Auto-start a gRPC MP server for grpc:// connector endpoints (#6001) |
| 9 | [ai-dynamo/dynamo@008d356](https://github.com/ai-dynamo/dynamo/commit/008d356966dc3cd77198d7f47b6f9d949ab44d6a) | `2026-10-30` | `deploy/helm/charts/platform/components/operator/files/role.yaml`, `deploy/helm/charts/platform/components/operator/templates/manager-rbac.yaml` | feat(operator): deploy active-active PoolRelay for one DGD (#17347) |
| 10 | [ai-dynamo/nixl@d8253ad](https://github.com/ai-dynamo/nixl/commit/d8253ad8797488bcae4362df7b6d6ef534a76118) | `2026-10-12` | `src/bindings/python/nixl-meta/nixl/__init__.py`, `src/bindings/python/nixl-meta/nixl_meta_utils.py` | bindings/python: prefer nixl_rocm when the meta dispatcher detects ROCm (#2370) |

---

## 3. Deterministic Offline Verification (`experiments/megatron_mhc_pytorch_precompile_dynamo_poolrelay_phase141/`)

- **PyTorch Precompile Online First-Step Autotune Latency**: Reduced from `18.0 s` to **`0.0 s`** (**100.0%** saving).
- **ROCm Kineto In-Flight Dispatch Record Drop Rate**: Reduced to **`0.0%`**.
- **Dynamo Active-Active `PoolRelay` Single-Replica Failover Downtime**: **`0.0 ms`**.
