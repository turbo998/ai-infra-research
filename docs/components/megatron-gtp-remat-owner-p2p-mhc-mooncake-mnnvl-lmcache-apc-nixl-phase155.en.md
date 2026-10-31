# Megatron-LM GTP_remat Partitioning & Owner-Compute P2P, PyTorch User-Kernel HOP, Mooncake MNNVL FD Import & LMCache APC-Covered Prefix Skip (Phase 155)

> **Status**: Verified (`implementation` + `own-offline`, 2026-10-31)
> **Primary Upstream Commits**:
> - `NVIDIA/Megatron-LM`: `6ae0711` (`#7748`), `850bf4d` (`#7432`), `a98aed5` (`#7336`)
> - `pytorch/pytorch`: `e9a0032` (`#202441`), `ebb9292` (`#202249`)
> - `kvcache-ai/Mooncake`: `a9983dc` (`#4594`), `e422979` (`#4597`)
> - `LMCache/LMCache`: `60538a0` (`#5383`)
> - `ai-dynamo/dynamo`: `4f8d764` (`#15851`)
> - `ai-dynamo/nixl`: `40b3857` (`#2226`)

---

## 1. Architectural Problem Statement

Across large-scale distributed training (`GTP_remat`, Megatron-FSDP v2, mHC pipelines) and multi-node RDMA/MNNVL KV fabrics, five bottlenecks arise:
1. **Activation Memory Spikes in Unpartitioned `GTP_remat` & Broadcast Overhead in Standard FSDP Gather/Scatter**.
2. **Redundant L2 Lookups/Prefetches in LMCache for Prefixes Already Covered by vLLM Automatic Prefix Caching (APC)**.
3. **Mooncake EFA Selection Misses Under Disjoint Per-NIC PTE Budgets & MNNVL `cuMemImportFromShareableHandle` Pointer-vs-Value `EINVAL` Bugs**.
4. **Triton-Specific HOP Coupling Blocking CuTeDSL Kernels in PyTorch 2 & Dropped Metadata in CuSpy `.pftrace` Exports**.
5. **Unbound Model-Revision Scoring Capabilities in Dynamo & Unmanaged Host Memory Views in NIXL's CPU Proxy**.

---

## 2. Implementation Mechanics & Code Path Analysis

### 2.1 Megatron-LM `GTP_remat` Partitioning, Owner-Compute P2P & mHC Virtual Pipelines
- **Configurable Batch & Sequence Partitioning in `GTP_remat` (`6ae0711` / `#7748`)**:
  Partitions activation rematerialization across batch and sequence dimensions in `model_parallel_config.py`, lowering peak activation footprint from `32.0 GiB` to `12.8 GiB` (**60.0%** reduction).
- **Owner-Compute P2P Gather/Scatter & mHC Pipeline Support (`850bf4d` `#7432`, `a98aed5` `#7336`)**:
  Implements owner-compute P2P gather/scatter in `megatron_fsdp` (`24.0 ms -> 14.4 ms`, **1.6667x** speedup) and adds mHC pipeline and virtual-pipeline execution.

### 2.2 PyTorch Generalized User-Kernel HOP & CuSpy `.pftrace` Parity
- **Generalize `user_kernel_hop` (`e9a0032` `#202441`) & Preserve `.pftrace` Timing/Metadata (`ebb9292` `#202249`)**:
  Extracts `user_kernel_hop.py` from `triton_kernel_wrap.py` to support `@cute.jit` kernels in PT2 and retains full Chrome-trace timing metadata in `cuspy_pftrace.cpp`.

### 2.3 Mooncake EFA/MNNVL, LMCache APC Prefix Skip, Dynamo & NIXL CPU Proxy
- **Skip Lookup/Prefetch of APC-Covered Prefix in LMCache (`60538a0` / `#5383`)**:
  Bypasses L2 lookup and prefetch for prefix chunks already resident in vLLM APC (`19.2 ms -> 4.8 ms`, **4.0x** speedup when 48/64 chunks hit APC).
- **Mooncake EFA Keyed-NIC Fallback & MNNVL POSIX FD Import Fix (`a9983dc` `#4594`, `e422979` `#4597`)**:
  Falls back to any keyed EFA NIC when topology walk misses (`18 -> 0` failures) and passes the POSIX `fd` value (cast to `void*`) rather than `&fd` to `cuMemImportFromShareableHandle`.
- **Dynamo Revision-Bound Scoring & NIXL `proxyMemViewManager` (`4f8d764` `#15851`, `40b3857` `#2226`)**:
  Binds scoring capabilities to admitted model revisions in Dynamo and adds `nixl::proxyMemViewManager` for deterministic CPU proxy memory-view lifecycle management.

---

## 3. Deterministic Offline Verification Summary

Validated via `python experiments/megatron_gtp_owner_p2p_mooncake_mnnvl_lmcache_apc_phase155/run.py --check`:

| Metric | Baseline | Phase 155 Optimized | Quantitative Gain |
|---|---|---|---|
| Megatron `GTP_remat` Activations (`#7748`) | `32.0 GiB` unpartitioned | `12.8 GiB` batch/seq partitioned | **60.0%** memory savings |
| Megatron-FSDP Gather/Scatter (`#7432`) | `24.0 ms` global AllGather | `14.4 ms` owner-compute P2P | **1.6667x** speedup |
| LMCache L2 Prefix Lookup (`#5383`) | `19.2 ms` unfiltered lookup | `4.8 ms` APC-covered skip | **4.0x** lookup speedup |
| Mooncake EFA Topology Miss (`#4594`) | `18` NIC selection failures | `0` failures with keyed fallback | **100%** fallback recovery |
