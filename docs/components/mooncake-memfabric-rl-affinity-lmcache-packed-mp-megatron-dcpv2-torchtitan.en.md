# Mooncake MemFabric Batch Registration & RL HCA Affinity, LMCache Packed uint32 Token Codec & Hybrid KV Groups, Megatron-LM DCP V2, and TorchTitan FA3/FA4 Verification

> 中文版: [mooncake-memfabric-rl-affinity-lmcache-packed-mp-megatron-dcpv2-torchtitan.md](mooncake-memfabric-rl-affinity-lmcache-packed-mp-megatron-dcpv2-torchtitan.md)

## 1. Context and Core Architectural Problems

Cross-node KV cache transport (Mooncake, LMCache) and large-scale distributed training (Megatron-LM, TorchTitan, TransformerEngine, Dynamo) face four infrastructure bottlenecks:

1. **NPU Hugepage Alignment Overlap in Per-Buffer Registration and Cross-NUMA RL Transfers (`Mooncake #4870, #4631`)**: Registering adjacent memory buffers one by one in MemFabric causes overlapping NPU page registrations after 2 MiB page alignment, while HCA-oblivious segment allocation in Mooncake Store routes RL `put`/`get` traffic across mismatched PCIe/NUMA domains.
2. **Python `tuple[int]` IPC Bloat and Multi-Group Hybrid KV Windowing (`LMCache #5150, #4410`)**: Passing token IDs across 3 IPC hops as `tuple[int, ...]` materializes `36 B/token` (`28 B` `PyLongObject` + `8 B` tuple pointer), while hybrid GDN + full-attention models require distinct trailing chunk windows per kernel group and padded SYCL block strides.
3. **Standalone Dynamic Context Parallelism (DCP) Scheduler Complexity and Checkpoint Integrity Loss (`Megatron-LM #7047, #6502`)**: Maintaining a separate 687-line `hybrid_cp_schedule.py` duplicated standard train/eval schedules, while overwriting `args.ckpt_format` before source checkpoint loading disabled `verify_integrity` during format conversion.
4. **Implicit Varlen Attention Fallback Risks and Lazy `attn_gym` Export Diagnostics (`TorchTitan #5546, #5565`)**: Running varlen attention on `SM90` (`FA3`) or `SM100` (`FA4`) without verifying `flash_attn_interface` or `flash_attn.cute` installation risks silent misconfiguration, while lazy `attn_gym` exports need per-symbol build failure isolation.

---

## 2. First-Hand Source Code Audit

### 2.1 Mooncake TENT `SmemTransBatchRegisterMem` and 3-Tier HCA Affinity (`Mooncake@6a835cbf934a`, #4870; `868a6d05abf9`, #4631)

- **Batch Memory Registration via `SmemTransBatchRegisterMem` (#4870)**: `MemFabricTransport::addMemoryBuffer(std::vector<BufferDesc>&, ...)` collects all buffer addresses and capacities into contiguous vectors and invokes `MemFabricUnderApi::SmemTransBatchRegisterMem(handle_, addrs.data(), caps.data(), count, 0)` once, allowing MemFabric to merge and align the batch to NPU pages without adjacent-buffer overlap errors.
- **Request-Local 3-Tier HCA Affinity in Mooncake Store (#4631)**: Under `MC_STORE_HCA_AFFINITY=1` and `MC_ENABLE_DEST_DEVICE_AFFINITY=1`, `RandomAllocationStrategy` and replica selection iterate across **3 affinity tiers**: Tier 0 (matching HCA-local ranges), Tier 1 (unknown locality segments), and Tier 2 (known mismatched HCA segments as a capacity fallback).

### 2.2 LMCache Big-Endian `uint32` Token Codec and Hybrid KV Layer Groups (`LMCache@6913259264db`, #5150; `00f6d71ed1af`, #4410)

- **Packed Big-Endian `uint32` `token_bytes` (#5150)**: `pack_token_ids` converts token sequences into `TOKEN_STRIDE = 4` B/token big-endian `uint32` buffers via `array.array("I")` + `byteswap()`, matching the exact byte layout consumed by `blake3` so chunk hashes are unchanged without unpacking. Across a `32,768`-token sequence, per-hop payload drops from `1,179,648 B` (`36 B/token`) to `131,072 B` (`4 B/token`), a **`9.0x`** reduction.
- **`KVLayerGroupsManager` Hybrid Window Chunks and SYCL Strides (#4410)**: `get_kernel_group_window_chunks` returns `-1` for full attention and $\max(1, \lceil \text{sw\_size\_tokens} / \text{chunk\_size} \rceil)$ for sliding-window or GDN recurrent groups; when `full_sw_kv=True`, sliding-window attention widens to `-1` while GDN recurrent state preserves its `1`-chunk window (`[-1, -1, 1]`). SYCL memory kernels also accept explicit `block_stride_elems` across `NL_X_NB_BSV_BSS`, `NL_X_NB_BS_NH_HS`, and fused-packed layouts.

### 2.3 Megatron-LM DCP V2 Schedule Integration and Checkpoint Integrity (`Megatron-LM@ea99fdfce50a`, #7047; `2052769557dc`, #6502)

- **DCP V2 Group Sizes and Schedule Unification (#7047)**: `get_valid_dynamic_context_parallel_group_sizes(total_size)` returns power-of-two divisors of `total_size` plus `total_size` itself (e.g., `total_size = 24 -> [1, 2, 4, 8, 24]`), integrating DCP V2 directly into standard train/eval schedules and deleting the 687-line `hybrid_cp_schedule.py`.
- **Source Checkpoint Integrity Preservation (#6502)**: Preserves `verify_integrity` when loading a `torch_dist` source checkpoint during `--ckpt-convert-format` runs before clearing `verify_integrity` for non-`torch_dist` target formats.

### 2.4 TorchTitan FA3/FA4 Verification, Exact FA2 Fallback, and `attn_gym` Isolation (`torchtitan@89e4339c963c`, #5546; `3ed5aa976fdf`, #5565)

- **Explicit `find_spec` Check for FA3/FA4 and Exact FA2 Fallback (#5546)**: Verifies `flash_attn.cute` (`FA4` on SM100+) and `flash_attn_interface` (`FA3` on SM90) via `importlib.util.find_spec`, while validating PyTorch SDPA FA2 fallback with `atol=0, rtol=0` (loss `0.19502778351306915`, grad L2 norm `0.012680011801421642`) and rejecting asymmetric MLA `QK != V` head dimensions.
- **`attn-gym==0.0.17` Per-Export Build Failure Isolation (#5565)**: Verifies 9 isolated `_BUILD_FAILURE_CASES` across KDA and GDN kernels so every entry in `require_attn_gym` is proven load-bearing.

---

## 3. Reproducible Verification & Summary Matrix

```bash
python experiments/mooncake_memfabric_rl_affinity_lmcache_packed_mp_megatron_dcpv2_torchtitan/run.py --check
python -m unittest tests/test_phase75_experiments.py
```

| Component / Repo | Mechanism / Function | Baseline / Pre-Fix Behavior | Post-Fix Quantitative Metric |
|---|---|---|---|
| Mooncake MemFabric (#4870) | `SmemTransBatchRegisterMem` | Per-buffer loop fails on 2 MiB NPU page overlap | **1 batch call**, 0 page-alignment overlap errors |
| Mooncake Store HCA Affinity (#4631) | 3-Tier `HcaAffinityTier` | Random segment allocation ignores HCA locality | **Tier 0 -> Tier 1 -> Tier 2** prioritized fallback |
| LMCache Token Codec (#5150) | Big-endian `uint32` `pack_token_ids` | `tuple[int]` costs `36 B/token` per IPC hop | **`4 B/token`** (**`9.0x`** reduction), direct Blake3 hashing |
| Megatron-LM DCP V2 (#7047) | `get_valid_dynamic_context_parallel_group_sizes` | 687-line standalone `hybrid_cp_schedule.py` | Removes **687 lines**; `total=24 -> [1, 2, 4, 8, 24]` |
| TorchTitan FA3/FA4 & `attn_gym` (#5546, #5565) | `find_spec` + 9 `_BUILD_FAILURE_CASES` | Unchecked missing FA3/FA4 wheels | SDPA FA2 `atol=0, rtol=0`; **9/9** export dependencies verified |
