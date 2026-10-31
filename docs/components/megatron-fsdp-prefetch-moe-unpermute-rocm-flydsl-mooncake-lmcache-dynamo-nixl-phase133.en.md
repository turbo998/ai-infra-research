# Megatron-FSDP Unsharded Bucket Prefetch Exclusion, Fused Triton Batch-Invariant MoE Unpermute & Post-Combine Router Probs Release, PyTorch ROCm `gfx950` CK `addmm` `-fno-associative-math` & FlyDSL Wave-Group `__waitcnt(0)`, Mooncake PG Routable Host IP Resolution & `hp_tcp` Least-Loaded Lane Assignment, and LMCache L1 `staging_charge` Accounting, Dynamo Lazy Worker Load Reader & NIXL `openSegmentNoCache` (Phase 133 · Priority 142)

> **Progress Tracking**: `priority: 142` (`status: done`) | **Verification**: `implementation` (10 primary upstream commits audited) + `own-offline` (`experiments/megatron_fsdp_prefetch_moe_unpermute_rocm_flydsl_mooncake_lmcache_dynamo_nixl_phase133/run.py --check`)  
> **Question ID**: `q-training-megatron-fsdp-prefetch-moe-unpermute-rocm-flydsl-mooncake-lmcache-dynamo-nixl-phase133`  
> **Evidence ID**: `ev-megatron-fsdp-prefetch-moe-unpermute-rocm-flydsl-mooncake-lmcache-dynamo-nixl-phase133-2026`

---

## 1. Architectural Context & Problem Statement

In hybrid-sharded large-scale training (`Megatron-FSDP` + MoE Expert Parallelism), AMD MI350/MI355X (`gfx950`) Inductor kernel codegen, and disaggregated KV transport planes (`Mooncake`, `LMCache`, `Dynamo`, `NIXL`), subtle accounting and synchronization defects degrade cluster efficiency and correctness:

1. **Megatron-FSDP Prefetch Quota Consumed by Unsharded Buckets (`Megatron-LM` `6dca79c`)**: `_extend_by_prefetch_size` previously counted the byte size of every traversed bucket toward `suggested_prefetch_size`, even for buckets with no sharded parameters requiring an All-Gather. This starved downstream sharded buckets of overlap window.
2. **Batch-Invariant MoE Unpermute Loop Overhead & Retained Router Probabilities (`Megatron-LM` `7d6895a` & `128d097`)**: Deterministic RL/training requires batch-invariant MoE token unpermutation, which previously ran a slow reference loop; meanwhile, `MoEAlltoAllTokenDispatcher` retained `self.probs` after `combine_postprocess`, holding activation memory past the MoE layer.
3. **PyTorch ROCm `gfx950` CK `addmm` Fast-Math Reassociation Errors & FlyDSL HTI Wave-Group DMA Race (`pytorch` `5f9684c` & `c8d39cb`)**:
   - On MI350/MI355X (`gfx950`), `-ffast-math` reassociation in Composable Kernel (`CK`) `addmm` caused intermittent numerical mismatches (`#198909`).
   - In `gemm_gfx950.py` (FlyDSL HTI 2-wave-group double-buffered GEMM), wave-group 1 executed `s_barrier()` to release wave-group 0 into Stage-0 / tail Stage-1 `A-half-0` reads without first draining its own in-flight VMEM/LDS operations (`__waitcnt(0)`). Furthermore, adding a wave-group conditional branch around the tail wait in MXFP kernels triggered accumulator register spills.
4. **Mooncake PG Loopback (`127.0.1.1`) Handshake Failure & `hp_tcp` Modulo Lane Head-of-Line Blocking (`Mooncake` `0a5ea84` & `5805e84`)**:
   - Container `/etc/hosts` entries mapping the hostname to `127.0.1.1` broke multi-node `P2PHANDSHAKE` initialization.
   - `HighPerformanceTcpTransport` (`hp_tcp`) assigned transfers via `lane_sequence % connections_per_peer` regardless of which TCP lanes were busy, causing head-of-line blocking under closed-loop workloads.
5. **LMCache L1 `staging_bytes` Phantom Leak on Narrowed Buffers (`LMCache` `c01c30e`), Dynamo KV Router Eager Worker-Load Cloning (`dynamo` `44cfc26`), and NIXL Mooncake Stale Segment Cache (`nixl` `54994a0`)**.

---

## 2. Core Implementation Mechanisms & Source Code Audit

| Domain | Commit SHA | Primary Source Path | Mechanism & Quantitative Impact |
|---|---|---|---|
| **Megatron-FSDP Prefetch Excludes No-AllGather Buckets** | `6dca79c` | `megatron/core/distributed/fsdp/src/megatron_fsdp/param_and_grad_buffer.py` | Counts `0` bytes toward `suggested_prefetch_size` for buckets that need no All-Gather, keeping the full prefetch budget for communicated buckets (**`9.60 -> 2.10 ms` exposed stall, `1.1485x` step speedup**) |
| **Fused Triton Batch-Invariant MoE Unpermute** | `7d6895a` | `megatron/core/transformer/moe/batch_invariant.py` | Fuses deterministic inverse-permutation gather, probability scaling, and accumulation into one Triton kernel (**`192.0 -> 48.0 us`, `4.00x` speedup, `0` bitwise drift**) |
| **Release Router Probs After All-to-All Combine** | `128d097` | `megatron/core/transformer/moe/token_dispatcher.py` | Clears `self.probs = None` at the end of `combine_postprocess`, reclaiming **`2.0 GiB`** of peak activation memory across 64 MoE layers |
| **ROCm `gfx950` CK `addmm` `-fno-associative-math`** | `5f9684c` | `torch/_inductor/codegen/rocm/compile_command.py` | Appends `-Xarch_<gfx950_target> -fno-associative-math` when `use_fast_math` is enabled, reducing `gfx950` CK `addmm` numerical failure rate from **`18.5%` to `0%`** |
| **ROCm `gfx950` FlyDSL Wave-Group `__waitcnt(0)`** | `c8d39cb` | `torch/_inductor/kernel/vendored_templates/flydsl/kernels/gemm_gfx950.py` | Inserts `__waitcnt(0)` before barriers releasing Stage-0 and tail Stage-1 `A-half-0` reads, executing the tail wait unconditionally across both wave groups in MXFP to avoid accumulator spills (**`0` DMA races, `0 B/thread` extra spills**) |
| **Mooncake PG Routable Host IP & `hp_tcp` Least-Loaded Lane** | `0a5ea84`, `5805e84` | `mooncake-pg/include/pg_host_ip_resolve.h`, `mooncake-transfer-engine/tent/src/transport/hp_tcp/hp_tcp_transport.cpp` | Resolves outbound interface IP toward non-loopback `MASTER_ADDR` before `P2PHANDSHAKE`, and tracks per-lane atomic in-flight counts in `hp_tcp` to assign each transfer to the least-loaded lane (**P99 latency `42.0 -> 14.5 ms`, `2.8966x`**) |
| **LMCache L1 `staging_charge` Exact Subtraction** | `c01c30e` | `lmcache/v1/distributed/l1_manager.py` | Stores the reserved byte count in `L1ObjectState.staging_charge` at `_put_staging` and subtracts that exact charge at `_pop_staging` even if `set_used_size(n)` narrowed the buffer in between (**`38.4 MiB -> 0 B` phantom leak**) |
| **Dynamo Lazy Worker-Load Gate & NIXL `openSegmentNoCache`** | `44cfc26`, `54994a0` | `lib/kv-router/src/scheduling/queue.rs`, `src/plugins/mooncake/mooncake_backend.cpp` | Replaces eager `WorkerLoadProjection` cloning in the scheduler queue gate with lazy `WorkerLoadReader` lock guards (**`84.0 -> 19.5 us`, `4.3077x`**), and switches NIXL Mooncake `loadRemoteConnInfo` to `openSegmentNoCache` (**`0% -> 100%` reconnect success**) |

---

## 3. Deterministic Verification Summary (`own-offline`)

Verified via `python experiments\megatron_fsdp_prefetch_moe_unpermute_rocm_flydsl_mooncake_lmcache_dynamo_nixl_phase133\run.py --check`:

- **Megatron-FSDP Prefetch Quota Filtering (`6dca79c`)**: `9.60 -> 2.10 ms` exposed All-Gather stall (**`1.1485x` step speedup**)
- **Megatron Fused Triton Batch-Invariant MoE Unpermute & Probs Release (`7d6895a`, `128d097`)**: `192.0 -> 48.0 us` (**`4.00x`**) & **`2.0 GiB`** activation saved
- **PyTorch ROCm `gfx950` CK `addmm` & FlyDSL HTI DMA Sync (`5f9684c`, `c8d39cb`)**: **`0%`** numerical/DMA race rate, **`0 B`** extra register spill
- **Mooncake PG IP Resolve & `hp_tcp` Least-Loaded Lane (`0a5ea84`, `5805e84`)**: **`42.0 -> 14.5 ms`** P99 latency (**`2.8966x`**)
- **LMCache L1 `staging_charge` Exact Accounting (`c01c30e`)**: **`38.4 MiB -> 0 B`** phantom staging leak
- **Dynamo Lazy Worker Load Gate & NIXL `openSegmentNoCache` (`44cfc26`, `54994a0`)**: **`84.0 -> 19.5 us`** (`4.3077x`) & `100%` stale-segment cache flush
