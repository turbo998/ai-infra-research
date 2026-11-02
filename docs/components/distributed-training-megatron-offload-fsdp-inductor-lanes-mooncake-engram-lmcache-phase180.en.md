# Distributed Training & KV/RDMA Storage: Megatron-LM Warmup/Backward Partial Activation Offload & Per-Group Step, PyTorch Inductor Live Reduction Lanes, ROCm HRR Snapshot Lock, Mooncake EngramStore & DMA-BUF 4GiB Wrap Guard (Phase 180)

> 🌐 **Language / 语言切换**: **[中文](distributed-training-megatron-offload-fsdp-inductor-lanes-mooncake-engram-lmcache-phase180.md)** | **[English](distributed-training-megatron-offload-fsdp-inductor-lanes-mooncake-engram-lmcache-phase180.en.md)**

---

## 1. Core Architectural Questions & Primary Source Evidence (Pillar 0 & Pillar 1)

Phase 180 (`priority: 189`) audits **10 primary upstream commits** across Megatron-LM, PyTorch Inductor, ROCm Systems, Mooncake, LMCache, and NVIDIA NIXL:

1. **Megatron-LM Warmup/Backward Partial Activation Offload, Unified FSDP ExecutionOrder & Per-Group Optimizer Step**:
   - `90f4d5279b95` (#7575) fixes partial fine-grained activation offloading (`--activation-offload-fraction` with `--offload-modules`) during pipeline warmup and backward passes in `megatron/core/pipeline_parallel/fine_grained_activation_offload.py`, delivering **`37.5%`** activation HBM savings with `0` warmup/backward D2H/H2D queue mismatches.
   - `2fcef8359b97` (#8485) unifies preset FSDP execution order traversal (`megatron_fsdp/experimental/execution_order.py`) around weak phase-tagged `ExecutionEvent` sequences with a sequential cursor (`100.0%` forward/backward prefetch traversal consistency).
   - `e0a4bfed7f1e` (#7158) tracks optimizer steps per parameter group (`param_group`, keyed by `pg_overrides`, `is_expert_parallel`, and `is_dsa_indexer`) instead of globally in `MegatronOptimizer` and `DistributedOptimizer` (`0` Adam bias-correction step skews on frozen/staged groups).
2. **PyTorch TorchInductor Live Reduction Lane Epilogues & FP32 Half-Precision Grouped Reductions**:
   - `4dd518df6250` (#200466) forwards live reduction tiles held in persistent-kernel registers directly into fixed-lane epilogues (e.g., `v.amax(-1) + v[:, 7]`) in `torch/_inductor/codegen/simd.py`, avoiding intermediate producer materialization and fusing producer + reduction + epilogue into a single kernel (`50.0%` memory traffic reduction, **`1.22x`** speedup).
   - `3da914e342b9` (#200463) uses `upcast_compute_type` (`float32`) for both reshape and reduction in `_GroupedReductionOpsHandler.reduction` (`simd.py`), eliminating invalid `<fp32 tile>.to(tl.int16, bitcast=True)` Triton compilation failures on `float16`/`bfloat16` grouped reductions (`0` bitcast compilation errors).
3. **ROCm Systems HIP Runtime Replay (`HRR`) Pinned Host Snapshot Mutex Guard**:
   - `9051cfb62f24` (#14346) acquires `ctx.host_free_mu` in shared mode (`t_launch_hold`) before a kernel launch yields the global replay turn and releases it once the launch call returns, while `hipHostFree`/`hipFreeHost`/`hipHostUnregister`/`hipFree` acquire `host_free_mu` exclusively (`0` pinned-host snapshot use-after-free races).
4. **Mooncake Unified `EngramStore` CPU/CUDA Lookup & DMA-BUF $2^{32}$ MR Wrap Guard + LMCache / NIXL Safety**:
   - `e33fe7b83c06` (#4701) adds the unified `EngramStore` table API (`mooncake-integration/store/engram_store_py.cpp`) supporting immutable opaque byte rows, persistent/anonymous local snapshots, synchronous CPU lookup, and stream-ordered CUDA Graph lookup (**`1.38x`** lookup speedup, `66.24 us -> 48.00 us`).
   - `03ff956700e8` (#5113) warns in `RdmaContext::registerMemoryRegionInternal` (`rdma_context.cpp`) when a DMA-BUF GPU Memory Region has a $2^k$-aligned start ($k \ge 32$, $\ge 4\text{ GiB}$) and length $> 2^k$, where Linux kernel `__rdma_block_iter_next()` 32-bit truncation causes RDMA transfers at offset $X$ to silently wrap to $X \bmod 2^k$ (`100.0%` wrap hazard detection).
   - `38ebdeac201d` (#4072) guards LMCache `TransferContext.close()` and adapter shutdown with `_transfer_ctx_lock` against in-flight gather/scatter operations (`0` close crashes); `328a3c195c27` (#2516) handles empty lists/tuples and zero-row 2D NumPy arrays in NIXL `get_reg_descs`/`get_xfer_descs`/`get_remote_descs` (`0` `IndexError`/UB faults).

---

## 2. Deterministic Offline Verification (Pillar 2 & Pillar 3)

Verified via [`experiments/megatron_offload_fsdp_inductor_lanes_mooncake_engram_lmcache_phase180/run.py`](../../experiments/megatron_offload_fsdp_inductor_lanes_mooncake_engram_lmcache_phase180/run.py) `--check`:
- `megatron_partial_offload_hbm_reduction_pct = 37.5`, `megatron_warmup_bwd_offload_mismatches = 0`
- `megatron_fsdp_preset_order_consistency_pct = 100.0`, `megatron_per_group_opt_step_skews = 0`
- `inductor_live_reduction_lane_epilogue_speedup = 1.22`, `inductor_half_grouped_reduction_bitcast_errors = 0`
- `rocm_hrr_pinned_snapshot_free_races = 0`
- `mooncake_engram_store_lookup_speedup = 1.38`, `mooncake_dmabuf_4gib_wrap_detection_pct = 100.0`
- `lmcache_inflight_gather_scatter_close_crashes = 0`, `nixl_empty_descriptor_index_errors = 0`

---

## 3. Enterprise Business Scenario Playbook (Pillar 4)

| Dimension | ✅ Good Fit | ⛔ Not For |
| :--- | :--- | :--- |
| **Long-Context Pipeline-Parallel & Staged Training** | Megatron-LM partial fine-grained activation offloading (`#7575`), unified phase-tagged FSDP execution order (`#8485`), and per-parameter-group optimizer step tracking (`#7158`). | Using a single global optimizer step counter across staged/frozen parameter groups (e.g., DSA indexer or frozen vision towers). |
| **Large-Buffer GPU DMA-BUF RDMA & Engram Tables** | Mooncake local `EngramStore` stream-ordered CUDA Graph lookups (`#4701`) and capping DMA-BUF MR sizes via `MC_MAX_MR_SIZE` with $2^{32}$-alignment wrap detection (`#5113`). | Registering unsplit $>4\text{ GiB}$ `cudaMalloc` allocations aligned to $2^{32}$ boundaries as a single DMA-BUF MR on unpatched Linux kernels (causes silent $X \bmod 2^k$ RDMA memory corruption). |
| **Graceful Multi-Process KV Offload & Compiler Fusion** | Fusing fixed-lane epilogues into persistent reductions in TorchInductor (`#200466`) and guarding LMCache context teardown (`#4072`) and NIXL empty descriptors (`#2516`). | Closing multi-process KV transfer contexts concurrently with active IPC gather/scatter ops without `_transfer_ctx_lock`. |
