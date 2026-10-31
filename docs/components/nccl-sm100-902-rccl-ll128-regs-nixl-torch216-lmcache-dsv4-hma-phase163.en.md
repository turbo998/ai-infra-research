# NCCL SM100 90.2 GB/s Search Restoration, RCCL LL128 Register Lane State & NCCL_RINGS Validation, NIXL Torch 2.16 & LMCache XPU DeepSeek-V4 HMA (Phase 163 · Priority 172)

> **中文版本**: [`nccl-sm100-902-rccl-ll128-regs-nixl-torch216-lmcache-dsv4-hma-phase163.md`](nccl-sm100-902-rccl-ll128-regs-nixl-torch216-lmcache-dsv4-hma-phase163.md)

## 1. Primary Source Evidence & Classification

| Source ID | Repository & Commit | Release Date | Evidence Tier | Key Code Path & Architectural Mechanism |
| :--- | :--- | :--- | :--- | :--- |
| `src-nccl-revert-sm100-search-902-to-451-2026` | `NVIDIA/nccl` (`acf5ddc01f`) | 2026-09-17 | `implementation` | `src/graph/search.cc`: Reverts accidental halving of SM100 graph search bandwidth (`45.1 -> 90.2`), restoring 2.0x channel search ceiling on Blackwell |
| `src-nccl-2322-version-release-2026` | `NVIDIA/nccl` (`198400a541`) | 2026-09-16 | `specification` | `makefiles/version.mk`: Releases `NCCL 2.32.2-1` version baseline |
| `src-rocm-rccl-ll128-lane-state-registers-2026` | `ROCm/rocm-systems` `#14128` (`59333762dc`) | 2026-10-31 | `implementation` | `projects/rccl/src/device/prims_ll128.h`: Keeps `LL128` per-lane state in VGPR registers, eliminating 4 scratch spills per loop iteration |
| `src-rocm-rccl-validate-nccl-rings-parsegraph-2026` | `ROCm/rocm-systems` `#13509` (`602daef311`) | 2026-10-31 | `implementation` | `projects/rccl/src/graph/rome_models.cc`: Validates user-supplied `NCCL_RINGS` topology strings in `parseGraph` |
| `src-rocm-rccl-gin-host-entry-tests-2026` | `ROCm/rocm-systems` `#13225` (`fe8e451e14`) | 2026-10-31 | `implementation` | `projects/rccl/test/host/JoinThreadsOnExit.h`: Covers every `gin_host.cc` entry point with deterministic host-only unit tests |
| `src-nixl-build-include-torch-216-wheels-2026` | `ai-dynamo/nixl` `#2537` (`e1c9302d0b`) | 2026-10-28 | `implementation` | `contrib/Dockerfile.manylinux`: Includes PyTorch `2.16` in NIXL EP wheel builds |
| `src-nixl-ci-seed-sanity-port-blocks-retry-2026` | `ai-dynamo/nixl` `#2481` (`06dd03bac4`) | 2026-10-28 | `implementation` | `.ci/jenkins/lib/build-wheel-matrix.yaml`: Seeds disjoint sanity port blocks per build and retries on port collisions |
| `src-mooncake-ci-build-only-tent-leg-targets-2026` | `kvcache-ai/Mooncake` `#5000` (`72b5cfe5de`) | 2026-10-28 | `implementation` | `.github/workflows/ci.yml`: Builds only the CMake targets required by each `tent-ci` leg |
| `src-lmcache-xpu-dsv4-hma-correctness-2026` | `LMCache/LMCache` `#6167` (`1f3d4f0c4e`) | 2026-10-30 | `implementation` | `.buildkite/k3_tests/multiprocess/scripts/workloads/vllm/hma-dsv4-correctness.sh`: Adds deterministic DeepSeek-V4 Hybrid Memory Architecture (HMA) correctness coverage on Intel XPU |

## 2. Deterministic Verification (`experiments/nccl_sm100_902_rccl_ll128_regs_nixl_torch216_lmcache_dsv4_hma_phase163`)

- **NCCL SM100 90.2 GB/s Search Restoration & RCCL LL128 Registers**: Restores **`2.0x`** SM100 search bandwidth (`45.1 -> 90.2 GB/s`) and eliminates all **4** scratch spills per `LL128` iteration (`-14.5%` small-message latency).
- **NIXL Seeded Port Blocks & LMCache XPU DeepSeek-V4 HMA**: Reduces concurrent NIXL CI port collision rate from **`12.5%`** to **`0.0%`** and verifies **`100.0%`** deterministic token match rate on LMCache XPU DeepSeek-V4 HMA.
