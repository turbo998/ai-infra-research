# Interconnect Datapath & Host Storage Backend Deep Dive (Phase 170): NCCL EFA-GDA `efa-dp-direct 1.1.0` (`backendVersion 2`), ROCm `hipFile` Host Backend & NIXL TSan Metadata Wait

> **Evidence Classification**: `implementation` (commit-pinned source audit) + `own-offline` (deterministic standard-library analytical verification)

---

## 1. Primary Upstream Commit Matrix

| Repository | Commit SHA | Date | Core Engineering Change |
|---|---|---|---|
| `NVIDIA/nccl` | `07dfce7930` | 2026-09-11 | `EFA GDA: request backendVersion 2` |
| `NVIDIA/nccl` | `6727ed4d60` | 2026-09-11 | `EFA GDA: vendor efa-dp-direct 1.1.0` |
| `ROCm/rocm-systems` | `6a167157e2` | 2026-10-30 | `feat(hipFile): Add new backend 'host' that handles read/write operations between files and registered host buffers (#10824)` |
| `ROCm/rocm-systems` | `ed0644d487` | 2026-10-30 | `fix(rocprofiler-sdk): Enable strict compiler warnings in CI builds (#14377)` |
| `ai-dynamo/nixl` | `21b6327b2e` | 2026-10-08 | `test: lengthen the md_manager remote-MD wait under sanitizers (#2349)` |
| `ai-dynamo/nixl` | `671760d4df` | 2026-10-09 | `Cleanups for OBJ S3 (#2356)` |
| `kvcache-ai/Mooncake` | `9fb270f7fc` | 2026-10-28 | `[CI/Build] Fix lint and formatting in distributed Store tests (#4982)` |
| `LMCache/LMCache` | `ad1d73891d` | 2026-10-29 | `[Docker] Include confluent-kafka in the vLLM and standalone images (#6147)` |

---

## 2. Architectural Analysis

### 2.1 NCCL EFA-GDA `efa-dp-direct 1.1.0` & `backendVersion 2` (`6727ed4d60`, `07dfce7930`)
- **144-Byte QP Layout & 128-Byte Inline RDMA Writes**: Refreshes vendored `amzn/efa-dp-direct` to `1.1.0` (`5b50aab8fa`) and resolves EFA-GDA in `gin_host.cc` to `backendVersion 2`, enabling the `144-byte` QP layout, runtime `64/128-byte` WQE geometry, `128-byte` inline RDMA writes, and `64-bit` request IDs.

### 2.2 ROCm `hipFile` Host Backend (`6a167157e2`) & NIXL TSan Wait (`21b6327b2e`)
- **`hipFile` `host` Backend for Registered CPU Buffers (`#10824`)**: Adds a `host` backend so `hipFile` transparently serves read/write operations on both GPU and CPU memory, eliminating runtime errors when the NIXL AIS plugin registers `DRAM_SEG` buffers (`0` errors).
- **NIXL Sanitizer Remote-MD Timeout (`#2349`)**: Extends `waitForRemoteMD` from `3s` to `15s` under ThreadSanitizer so `etcd` invalidation reaches remote peers without false failures (`0` timeouts).

---

## 3. Deterministic Offline Verification

Verified via `python experiments/nccl_efa_dp_direct110_hipfile_host_nixl_tsan_phase170/run.py --check`:
- `efa_gda_backend_version`: **`2`** (`efa_dp_direct_qp_bytes = 144`, `efa_inline_rdma_write_bytes = 128`, `efa_req_id_bits = 64`)
- `host_backend_dram_seg_errors`: **`0`** (`host_backend_supported_mem_types = 2`)
- `tsan_etcd_invalidation_timeouts`: **`0`** (`sanitized_tsan_md_wait_s = 15`)
