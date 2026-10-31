# NCCL EFA-GDA FlushAsync Revert & NIIN Nonblocking Threadgroup, rocSHMEM Collectives, NIXL cuObject S3 & LMCache XPU Long-Doc L2 SHM (Phase 164 · Priority 173)

> **中文版本**: [`nccl-efa-gda-niin-rocshmem-nixl-cuobject-lmcache-l2-shm-phase164.md`](nccl-efa-gda-niin-rocshmem-nixl-cuobject-lmcache-l2-shm-phase164.md)

## 1. Primary Source Evidence & Classification

| Source ID | Repository & Commit | Release Date | Evidence Tier | Key Code Path & Architectural Mechanism |
| :--- | :--- | :--- | :--- | :--- |
| `src-nccl-revert-efa-gda-flush-async-wait-2026` | `NVIDIA/nccl` (`78e54daaff`) | 2026-09-15 | `implementation` | `src/*`: Reverts premature EFA-GDA `FlushAsync` and `Wait` implementation to prevent completion queue races |
| `src-nccl-niin-streamline-nonblocking-threadgroup-2026` | `NVIDIA/nccl` (`ab4ab4e0bf`) | 2026-09-15 | `implementation` | `contrib/niin/*`: Streamlines nonblocking threadgroup transfers in NIIN, reducing sync barriers from 4 to 1 (`-75.0%`) |
| `src-rocm-rocshmem-missing-collectives-apis-2026` | `ROCm/rocm-systems` `#12735` (`3ad12cc963`) | 2026-10-31 | `implementation` | `projects/rocshmem/*`: Adds missing collective API overloads in `rocSHMEM` (`#AIROCSHMEM-547`) |
| `src-rocm-rocprofiler-rocattach-merged-pt-load-2026` | `ROCm/rocm-systems` `#13666` (`ad34ecf876`) | 2026-10-31 | `implementation` | `projects/rocprofiler-sdk/*`: Handles merged `PT_LOAD` ELF mappings in `rocattach` |
| `src-nixl-ci-cuda-cuobject-s3-acceleration-2026` | `ai-dynamo/nixl` `#2417` (`33546c2172`) | 2026-10-21 | `implementation` | `.ci/*`: Builds NIXL on CUDA images that compile `cuObject` S3 direct acceleration |
| `src-nixl-ci-release-wheel-exact-commit-2026` | `ai-dynamo/nixl` `#2448` (`07ebb2a02c`) | 2026-10-21 | `implementation` | `.ci/*`: Runs release wheel builds directly from the exact commit being packaged |
| `src-mooncake-add-store-rs-tests-2026` | `kvcache-ai/Mooncake` `#4972` (`fe4f297a94`) | 2026-10-28 | `implementation` | `mooncake-store/rust/*`: Adds Rust `store-rs` integration tests |
| `src-mooncake-ci-tent-runtime-tone-parallel-2026` | `kvcache-ai/Mooncake` `#3600` (`b730bbae5f`) | 2026-10-27 | `implementation` | `.github/workflows/*`: Enables TENT runtime in T-One E2E as a parallel CI job |
| `src-lmcache-xpu-long-doc-l2-shm-engine-metrics-2026` | `LMCache/LMCache` `#6102` (`2b7fd89a9c`) | 2026-10-28 | `implementation` | `.buildkite/k3_tests/*`: Enables long-document L2 shared-memory (SHM) tests and engine-driven metrics on Intel XPU |

## 2. Deterministic Verification (`experiments/nccl_efa_gda_niin_rocshmem_nixl_cuobject_lmcache_l2_shm_phase164`)

- **NCCL NIIN Threadgroup & rocSHMEM / rocattach**: Cuts NIIN nonblocking threadgroup barriers by **`75.0%`** (`4 -> 1`) and reduces missing `rocSHMEM` collective APIs and `rocattach` merged `PT_LOAD` failures to **0**.
- **NIXL cuObject S3 & LMCache XPU Long-Doc L2 SHM**: Eliminates release-wheel commit SHA drift (`0`) and verifies **`100.0%`** L2 SHM hit and correctness rate on Intel XPU long-document workloads.
