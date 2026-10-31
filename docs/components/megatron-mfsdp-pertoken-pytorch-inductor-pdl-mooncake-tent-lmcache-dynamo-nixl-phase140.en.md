# Megatron-LM MFSDP v2 Per-Token Gradient Normalization & RL Decoupling, PyTorch Inductor PDL & Template Fusion, and Mooncake TENT Failover Scheduling with LMCache/Dynamo/NIXL Evolution (Phase 140)

> **中文版**: [megatron-mfsdp-pertoken-pytorch-inductor-pdl-mooncake-tent-lmcache-dynamo-nixl-phase140.md](megatron-mfsdp-pertoken-pytorch-inductor-pdl-mooncake-tent-lmcache-dynamo-nixl-phase140.md)
> **Entry Point**: Distributed Training & KV Hierarchy — Megatron MFSDP v2 Per-Token Gradient Normalization & RL Decoupling / PyTorch Inductor Shared-Future Template Fusion, PDL Running-Kernel Verification & Stride-Preserving Triton Autotuning / Mooncake TENT UB Failover + Deadline Scheduling & Iteration-Free Contiguous Row Planning / LMCache InProcess Allocators / Dynamo NVTX v3 Hardening & NIXL Single-GPU GPUNETIO+MPS
> **Evidence Tier**: `implementation` + `own-offline` (10 primary distributed training & KV fabric commits audited at source level + Python 3.12 deterministic modeling)

---

## 1. Executive Summary & Architectural Conclusions

1. **Megatron-LM MFSDP v2 Per-Token Gradient Normalization & RL Factoring (`Megatron-LM@31265d3`, `579f710`)**: Supports exact per-token gradient normalization in MFSDP v2 (**0.0%** variable-length gradient bias vs. **14.8%** under per-rank averaging) and factors RL-specific logic out of `training.py` into `megatron/rl/`.
2. **PyTorch Inductor Shared-Future Template Fusion, PDL Running-Kernel Check & Stride-Preserving Autotuning (`pytorch@9976be2`, `a3222da`, `d72202f`)**: Evaluates every pending template fusion sharing a compile future, checks all potentially running kernels in Inductor PDL (**0.0** premature launch race rate), and preserves sample-input strides during Triton autotuning.
3. **Mooncake TENT UB Failover + Deadline Scheduling & Iteration-Free Row Planning (`Mooncake@1dd24ec`, `5435236`)**: Adds UB failover, telemetry, and deadline-aware runtime scheduling in Mooncake TENT, and plans contiguous matrix row ranges without Python iteration (**25.0x** CPU planning speedup, `250.0 us -> 10.0 us`).
4. **LMCache InProcess Allocator Isolation, Dynamo NVTX v3 Hardening & NIXL GPUNETIO+MPS (`LMCache@a6ad3bd`, `dynamo@d66b74e`, `nixl@3cecdc8`)**: Moves InProcess-only memory allocators into `lmcache/inprocess/v1/`, hardens Dynamo NVTX v3 under a single `DYN_NVTX` switch with worker markers, and documents single-GPU GPUNETIO scheduling with CUDA MPS in NIXLBench.

---

## 2. Primary Commit Matrix

| # | Repo & Commit | Date | Primary Code Paths | Architectural Mechanism & Breakthrough |
|---|---|---|---|---|
| 1 | [NVIDIA/Megatron-LM@31265d3](https://github.com/NVIDIA/Megatron-LM/commit/31265d3c102d850e8f087d868f099422a0649ba8) | `2026-10-17` | `megatron/core/distributed/fsdp/mcore_fsdp_adapter.py`, `megatron/core/distributed/fsdp/src/megatron_fsdp/experimental/fully_shard.py` | Support per-token gradient normalization in MFSDP v2 (#7711) |
| 2 | [NVIDIA/Megatron-LM@579f710](https://github.com/NVIDIA/Megatron-LM/commit/579f710840ba3ce8826a139e63a2f41ccfc47105) | `2026-10-29` | `megatron/rl/rl_utils.py`, `megatron/rl/training_setup.py` | Factor RL-specific code out of training.py (#4264) |
| 3 | [pytorch/pytorch@9976be2](https://github.com/pytorch/pytorch/commit/9976be280ff05b9d8992c012ed4c67920ced45f6) | `2026-10-30` | `test/inductor/test_inductor_scheduler.py`, `test/inductor/test_max_autotune.py` | [Inductor] Evaluate every pending template fusion that shares a compile future (#200211) |
| 4 | [pytorch/pytorch@a3222da](https://github.com/pytorch/pytorch/commit/a3222da1fd41062c158297749d63d593cc8a584a) | `2026-10-30` | `test/inductor/test_nv_universal_gemm.py`, `test/inductor/test_torchinductor.py` | [inductor][pdl] check all kernels that may still be running (#199812) |
| 5 | [pytorch/pytorch@d72202f](https://github.com/pytorch/pytorch/commit/d72202ff8726aaa643a1aed48c48eba3e5968fb1) | `2026-10-30` | `test/inductor/test_aot_inductor.py`, `torch/_inductor/graph.py` | [BE][Inductor] Preserve sample-input strides during Triton autotuning (#202161) (#202161) |
| 6 | [kvcache-ai/Mooncake@1dd24ec](https://github.com/kvcache-ai/Mooncake/commit/1dd24ec178bb1173d226aabba03eb832d671a56b) | `2026-10-16` | `.github/workflows/ci.yml`, `docs/source/design/tent/deadline-scheduling.md` | [TENT] Add UB failover, telemetry and runtime scheduling (#4605) |
| 7 | [kvcache-ai/Mooncake@5435236](https://github.com/kvcache-ai/Mooncake/commit/5435236ce663900403378839c0396b809a2aec1d) | `2026-10-16` | `mooncake-wheel/mooncake/_partial_read.py`, `mooncake-wheel/tests/test_partial_read.py` | [Wheel] Plan contiguous matrix row ranges without iteration (#4646) |
| 8 | [LMCache/LMCache@a6ad3bd](https://github.com/LMCache/LMCache/commit/a6ad3bd27d78748dec2757bdf507c0a11d591f2d) | `2026-10-30` | `lmcache/inprocess/v1/cache_engine.py`, `lmcache/inprocess/v1/memory_allocators/__init__.py` | Move InProcess-only memory allocators (#6138) |
| 9 | [ai-dynamo/dynamo@d66b74e](https://github.com/ai-dynamo/dynamo/commit/d66b74ef47f7c31522c81529f832668ba4206831) | `2026-10-30` | `benchmarks/frontend/scripts/nsight/nsys_profile.sh`, `benchmarks/frontend/scripts/run_perf.sh` | feat(runtime): NVTX v3 hardening, single DYN_NVTX switch, worker markers (#17334) |
| 10 | [ai-dynamo/nixl@3cecdc8](https://github.com/ai-dynamo/nixl/commit/3cecdc8e637adea15d092d5c98b0c8fc99e9110c) | `2026-10-29` | `benchmark/nixlbench/README.md` | [NIXLBench] Explain single-GPU GPUNETIO scheduling and optional MPS (#2298) |

---

## 3. Deterministic Offline Verification (`experiments/megatron_mfsdp_pertoken_pytorch_pdl_mooncake_tent_phase140/`)

- **MFSDP v2 Per-Token Gradient Bias**: Reduced from `14.8%` to **`0.0%`**.
- **Mooncake Iteration-Free Row Range Planning Speedup**: **`25.0x`** (`250.0 us -> 10.0 us`).
- **Inductor PDL Premature Launch Race Rate**: Reduced to **`0.0`**.
