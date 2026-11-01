# Distributed Training & Interconnect: Megatron MFSDP v2 Outer-Stream Overlap & MXFP8 Dual-Plane Unshard, TE THD CP Shared-Memory Guard, NCCL P2P Profiler & Mooncake URMA (Phase 179)

> 🌐 **Language / 语言切换**: **[中文](distributed-training-megatron-mfsdp-v2-mxfp8-te-thd-cp-nccl-mooncake-urma-phase179.md)** | **[English](distributed-training-megatron-mfsdp-v2-mxfp8-te-thd-cp-nccl-mooncake-urma-phase179.en.md)**

---

## 1. Core Architectural Questions & Primary Source Evidence (Pillar 0 & Pillar 1)

Phase 179 (`priority: 188`) audits **11 primary upstream commits** across Megatron-LM MFSDP v2, TransformerEngine, NCCL, Mooncake, LMCache, and NIXL:

1. **Megatron-LM MFSDP v2 Dedicated DP-Outer Reduction Stream, MXFP8 Dual-Plane Unshard & Recorded Prefetch Order**:
   - `037b883c9e79` (#6973) overlaps MFSDP v2 DP-outer gradient reduction on its own dedicated CUDA stream (`mfsdp_dp_outer_comm.py`, `param_and_grad_buffer.py`), hiding **`84.0%`** of inter-node outer-DP ReduceScatter latency (`25.0 ms -> 4.0 ms`) and yielding a **`1.18x`** step speedup.
   - `88312648a9dd` (#8432) adds `SchedulePolicy.unshard_all_planes_in_backward` to unshard both primary and transposed weight planes during MXFP8 activation recomputation (`mfsdp_v2_mxfp8_dual_plane_unshard_verified = True`).
   - `2d7ea246f936` (#8478) records dynamic MFSDP module execution order to drive exact next-step backward prefetching (`100.0%` order match).
2. **TransformerEngine THD Context-Parallel (CP) Shared-Memory-Aware Kernels & Scale Swizzling**:
   - `e619cea17e6f` (#3698) makes PyTorch `THD` CP helper kernels shared-memory-size aware, eliminating `cudaErrorInvalidValue` launch failures on large-head/high-CP shapes (`0` SMEM overflow errors).
   - `e614b7a9d3c3` (#3682) optimizes block-scale swizzling kernels and removes obsolete narrow scaled kernels (**`1.15x`** swizzle speedup).
3. **NCCL Bidirectional P2P `KernelCh` Profiler & Mooncake Multi-Context URMA Lifetime**:
   - `5c1b628eca9c` and `352fb9471288` honor `KernelCh` from either P2P direction and read the profiling bit at the batch stride (`0` dropped P2P profiler events).
   - `03496197d71d` (#5104) keeps the Mooncake TransferEngine `URMA` module alive across multiple Unified Bus (UB) contexts (`0` teardown crashes), while `0efca356cbc9` (#4959) strips debug symbols from wheels and enforces size limits (**`72.5%`** size reduction, `145.6 MB -> 40.04 MB`).
   - `08779cc9b5e4` (#6043) removes legacy SGLang KV format discovery in LMCache, and `641e24584ced` (#2440) allocates NIXL GPUNETIO `nixlbench` OOB ports from the reserved port range (`0` collisions).

---

## 2. Deterministic Offline Verification (Pillar 2 & Pillar 3)

Verified via [`experiments/megatron_mfsdp_v2_outer_stream_mxfp8_te_thd_cp_nccl_mooncake_urma_phase179/run.py`](../../experiments/megatron_mfsdp_v2_outer_stream_mxfp8_te_thd_cp_nccl_mooncake_urma_phase179/run.py) `--check`:
- `mfsdp_v2_outer_dp_overlap_ratio_pct = 84.0`
- `mfsdp_v2_step_speedup = 1.18`
- `mfsdp_v2_mxfp8_dual_plane_unshard_verified = True`
- `mfsdp_prefetch_execution_order_accuracy_pct = 100.0`
- `te_thd_cp_smem_overflow_launches = 0`
- `te_scale_swizzle_speedup = 1.15`
- `nccl_p2p_profiler_dropped_events = 0`
- `mooncake_multi_ub_context_urma_crashes = 0`
- `mooncake_wheel_size_reduction_pct = 72.5`

---

## 3. Enterprise Business Scenario Playbook (Pillar 4)

| Dimension | ✅ Good Fit | ⛔ Not For |
| :--- | :--- | :--- |
| **Hybrid-Sharded Pre-Training (Megatron MFSDP v2 + MXFP8)** | Overlapping DP-outer reductions on a dedicated stream (`84.0%` hidden, `#6973`), enabling `unshard_all_planes_in_backward` (`#8432`), and recording execution order for prefetch (`#8478`). | Unsharding only a single weight plane during MXFP8 recomputation or assuming static layer order in dynamic MoE/skip graphs. |
| **Variable-Length THD Context Parallelism (CP)** | Deploying shared-memory-aware TE THD CP helper kernels (`#3698`) and optimized scale swizzling (`#3682`). | Hardcoding static shared-memory allocations that exceed per-SM limits on wide-head CP workloads. |
| **Multi-UB KV Fabrics & Collective Telemetry** | Retaining URMA module lifetime across UB contexts in Mooncake (`#5104`), bidirectional P2P `KernelCh` profiling in NCCL (`5c1b628eca9c`), and reserved OOB ports in NIXL (`#2440`). | Unloading URMA shared objects when the first UB context closes or shipping unstripped 145MB+ wheels (`#4959`). |
