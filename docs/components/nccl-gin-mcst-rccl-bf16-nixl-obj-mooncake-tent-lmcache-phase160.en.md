# NCCL GIN Request Aggregation & GDAKI MCST, RCCL BF16 MinMax & Direct AG, NIXL Sender Progress & OBJ Batching, Mooncake TENT/EP Guards & LMCache Inprocess Architecture (Phase 160)

> **Evidence Class**: `implementation` (primary upstream commit audit) + `own-offline` (deterministic standard-library experiment `experiments/nccl_gin_mcst_rccl_bf16_nixl_obj_mooncake_tent_lmcache_phase160/`)

## 1. Primary Upstream Commit Matrix (Track 3: 10 Commits)

| # | Repo & Commit | Date | Key Implementation Path & Source Location |
|---|---|---|---|
| 1 | `NVIDIA/nccl@62b72dc38a` | 2026-09-27 | `docs/userguide/source/api/device_gin.rst`, `flags.rst`: Documents `ncclGinOptFlagsAggregateRequests` and rail/full connection semantics for device-initiated GIN operations. |
| 2 | `NVIDIA/nccl@10f2296bac` | 2026-09-15 | `src/transport/net_ib/gdaki/gin_host_gdaki.cc`: Adds `NCCL_GDAKI_FORCE_MCST` (`GinGdakiForceMcst`) to force hardware multicast during GDAKI QP setup. |
| 3 | `ROCm/rccl@fa366ac03f` | 2026-01-20 | `src/device/reduce_kernel.h`: Adds `SPECIALIZE_REDUCE(FuncMinMax, __nv_bfloat16, ...)` using `__hmin/__hmax` to fix broken bitwise integer comparisons on negative BF16 values in ROCm >= 6.0. |
| 4 | `ROCm/rccl@f3c5156bbf` | 2026-01-20 | `src/collectives.cc`: Reverts redundant `cudaMemcpyAsync` self-copy in direct `ncclAllGather_impl`. |
| 5 | `ai-dynamo/nixl@749442d4e7` | 2026-10-07 | `test/gtest/test_transfer.cpp`: Progresses the sender's UCX worker inside `verifyNotifs` retry loops so queued notifications do not stall when no background progress thread is active. |
| 6 | `ai-dynamo/nixl@6c5edaffbf` | 2026-10-06 | `benchmark/nixlbench/src/worker/nixl/nixl_worker.cpp`: Fixes `device-ID -> object name` collisions in `nixlbench` OBJ benchmarks when `batch_size > 1`. |
| 7 | `kvcache-ai/Mooncake@dd21db78b5` | 2026-10-15 | `mooncake-transfer-engine/tent/src/runtime/transfer_engine_impl.cpp`: Rejects `MC_TENT_CONF` configuration file load failures before TENT runtime startup. |
| 8 | `kvcache-ai/Mooncake@ba6d44d962` | 2026-10-14 | `mooncake-ep/src/mooncake_ep_buffer.cpp`: Validates legacy EP buffer constructor arguments to prevent workspace buffer overflows on underestimated dispatch token counts. |
| 9 | `LMCache/LMCache@4159596c25` | 2026-10-21 | `lmcache/inprocess/integration/vllm/vllm_service_factory.py`: Moves `HealthMonitor` under `lmcache.inprocess.v1.health_monitor`. |
| 10 | `LMCache/LMCache@45e928595f` | 2026-10-21 | `lmcache/inprocess/integration/vllm/vllm_v1_adapter.py`: Consolidates async and Mooncake lookup clients under `lmcache.inprocess.v1.lookup_client`. |

## 2. Key Implementation Mechanics & Formulas

### 2.1 NCCL GIN Request Aggregation & RCCL BF16 MinMax Specialization
Aggregating $N_{\text{put}}=64$ small device-initiated GIN puts with batch factor $B=8$ (`ncclGinOptFlagsAggregateRequests`) cuts NIC doorbell MMIO writes from `64` to `8` (**$8.0\times$** reduction). Meanwhile, specializing `FuncMinMax` for `bfloat16` (`ROCm/rccl@fa366ac03f`) replaces integer bitwise comparison with `__hmin/__hmax`, eliminating all `19` negative-number ordering errors.

### 2.2 NIXL Multi-Batch Object Naming & UCX Sender Progression
With `devices = 4` and `batch_size = 8` (`32` total objects), keying object names solely by `device_id` caused $4 \times 7 = 28$ overwrite collisions (`ai-dynamo/nixl@6c5edaffbf`); incorporating batch index eliminates all collisions (`28 -> 0`), while explicit sender worker polling (`ai-dynamo/nixl@749442d4e7`) eliminates TSAN notification timeouts (`4 -> 0`).

## 3. Deterministic Offline Verification (`own-offline`)

Validated via `python experiments/nccl_gin_mcst_rccl_bf16_nixl_obj_mooncake_tent_lmcache_phase160/run.py --check`:
- **NCCL GIN Doorbell Reduction**: `8.0x` (`64 -> 8`), RCCL BF16 MinMax errors `19 -> 0`.
- **NIXL Object Collisions & Notification Timeouts**: Collisions `28 -> 0`, TSAN notification timeouts `4 -> 0`.
- **Mooncake TENT/EP Startup Guards & LMCache Inprocess**: Silent invalid-config starts `3 -> 0`, EP buffer overflows `0`, inprocess lookup hop reduction `185.0 us`.
