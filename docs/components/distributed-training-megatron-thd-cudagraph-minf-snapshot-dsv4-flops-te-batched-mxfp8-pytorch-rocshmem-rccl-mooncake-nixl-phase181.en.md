# Distributed Training & Collective Fabrics: Megatron-LM THD Sequence-Packed Attention CUDA Graphs & MInf Token Snapshotting, TE Batched MXFP8 GEMM, PyTorch Triton `rocSHMEM` Device Collectives, RCCL Zen CPUID Topology Fix & Mooncake/LMCache/NIXL Hardening (Phase 181)

> 🌐 **Language / 语言切换**: **[中文](distributed-training-megatron-thd-cudagraph-minf-snapshot-dsv4-flops-te-batched-mxfp8-pytorch-rocshmem-rccl-mooncake-nixl-phase181.md)** | **[English](distributed-training-megatron-thd-cudagraph-minf-snapshot-dsv4-flops-te-batched-mxfp8-pytorch-rocshmem-rccl-mooncake-nixl-phase181.en.md)**

---

## 1. Core Architectural Problems & Primary Source Evidence (Pillar 0 & Pillar 1)

Phase 181 (`priority: 190`) audits **9 groups (11 commits) of upstream mainline changes** across Megatron-LM, TransformerEngine, PyTorch SymmetricMemory, ROCm RCCL/ROCr, Mooncake, LMCache, and NIXL:

1. **Megatron-LM `dp_balanced` THD Sequence-Packed TE Attention CUDA Graphs, MInf Token-Level Live Request Snapshotting for RL, DSv4 HybridModel Attention FLOPs & Consumer-Stream Storage Recording**:
   - `631e146322af` (#8361) enables Transformer Engine attention CUDA graph capture and replay for THD batches produced by `dp_balanced` sequence packing, preserving changing document boundaries, all four logical/physical Q/KV offset tensors, and changing microbatch counts (`4 -> 2 -> 4`) with fixed-shape token/metadata buffers, passing **`288/288`** output and gradient checks on 4x GB200 (`PP2/CP2` and `PP2/VPP2/CP2`, including fused CSA ratio-4 with sparse indexer loss and HCA ratio-128).
   - `01fa4164c8b5` (#8137) adds `DynamicInferenceEngine.snapshot_live_requests()` (`LiveRequestSnapshot`) and `ResumedGeneration` in Megatron Inference (`MInf`) so interrupted RL rollouts can save and restore live requests at token-level granularity (`0` lost in-flight tokens).
   - `e1afbb93db5c` (#7972) counts DeepSeek-V4 C/H/W attention layers (Q/KV/O projections, window attention, compression, ratio-4 indexer and sparse attention, ratio-128 HCA, and MTP depth) in `HybridModel` training FLOPs (`48/48` CPU test cases passed); `a0ef64615cc3` (#8823) records every `ScheduleNode` tensor input on the consumer stream regardless of `free_input`, preventing producer-stream allocations from overwriting retained inputs during asynchronous backward passes.
2. **TransformerEngine `BatchedLinear` BF16/MXFP8 Strided Batched GEMM & `AdamCapturableFunctor` Load/Store Ordering**:
   - `b021980343e3` (#3160) adds strided batched GEMM and `ops.BatchedLinear` in BF16 and MXFP8, inferring layout from per-tensor batch dimensions and falling back to high-precision GEMM for non-MXFP8 quantization formats.
   - `5c3f067c28b6` (#3782) optimizes `AdamCapturableFunctor` by reading `lr` and `inv_scale` once, issuing all loads before writing the unscaled gradient, and skipping that write when `inv_scale == 1` (`0` redundant unscaled writes).
3. **PyTorch Triton `rocSHMEM` Device-Side `alltoall`/`broadcast` & ROCm RCCL Zen 2/3/4 CPUID Topology Restoration / ROCr SVM Pointer Identification**:
   - `fa6060d66f9d` (#202454) enables device-side `alltoall` and `broadcast` (`2` collectives) in PyTorch SymmetricMemory's `rocSHMEM` Triton backend via an LLVM IR bitcode wrapper that injects `rocshmem_ctx_t` and `rocshmem_team_t` while exposing integer-only parameters to Triton.
   - `b1a305f97a6a` (#14146) restores Rome-class detection (`rcclTopoCpuIsRome()`) on Zen 2/3/4 (`family 0x17/0x19`, CPUID `23/25`) hosts after the NCCL 2.31.2 sync changed topology XML family IDs from `143/175/191` to `23/25/26`, recovering `parseRome4P2H`, `parse1H16P`, `parse4H4P`, `RCCL_TOPO_4P2H_ROME` ring marking, and same-root-complex GPU+NIC **`GDR PHB`** relaxation (`5` consumers); `34442d5ccd80` (#14694) updates ROCr `hsa_amd_pointer_info` to report `HSA_EXT_POINTER_TYPE_SVM (7)` after `hsa_amd_svm_attributes_set`.
4. **Mooncake XPU IPC Exporter `pidfd` Cleanup & DFS LRU Key Deduplication, LMCache Non-Blocking ZMQ & NIXL Range-Overflow / Disconnect Fixes**:
   - Mooncake `e5ef0e427a49` (#5026) watches Level Zero IPC exporter processes with `pidfd` in a dedicated thread and closes imported mappings as soon as the exporter exits (`0` leaked VRAM bytes across peer restarts); `e8dda21b8fab` (#4787) keys `ShardAllocator`'s DFS LRU `unordered_map` index with `std::string_view` into address-stable `std::list` nodes, cutting key copies from `2` to `1` (**`50%`** reduction).
   - LMCache `719f1d643558` (#6256) switches ZMQ request clients to `NOBLOCK` + `POLLOUT` polling so a blocked/down server (1,100 queued requests) cannot freeze healthy clients (`0` head-of-line stalls), and `236f3eec32a2` (#6208) extracts `PeerL2AdapterBase` with retryable teardown; NIXL `a91561a83f81` (#2142) makes `nixlBasicDesc::covers()` overflow-safe for `SIZE_MAX` unbounded file registrations (`unbounded_covers_finite=1, finite_covers_wrapping=0`), and `c656976fcd6d` (#2560) disconnects `remoteBackends_` on `loadRemoteSections` failure so subsequent `loadRemoteMD` calls succeed.

---

## 2. End-to-End Dataflow & Architecture Topology (Pillar 1)

```
+------------------------------------------------------------------------------------------------------+
| Phase 181 Distributed Training, Collectives & KV Fabrics: Megatron, TE, PyTorch, RCCL, Mooncake, NIXL|
+------------------------------------------------------------------------------------------------------+
  [Megatron-LM & TransformerEngine Training / RL Rollout Stack (#8361, #8137, #7972, #8823, #3160, #3782)]
    |-- Megatron THD CUDA Graph (#8361) --> dp_balanced Packed TE Attention Graph (288/288 GB200 Checks)
    |-- Megatron MInf Snapshot (#8137)  --> Token-Level LiveRequestSnapshot + ResumedGeneration for RL
    |-- Megatron DSv4 & PP (#7972/#8823)--> C/H/W+MTP FLOPs (48/48) + Unconditional Consumer Stream Record
    +-- TE Batched GEMM & Adam (#3160/#3782)-> BF16/MXFP8 BatchedLinear + Capturable Adam Load-Before-Store
                                      |
                                      v
  [PyTorch rocSHMEM, ROCm RCCL/ROCr, Mooncake, LMCache & NIXL (#202454, #14146, #5026, #4787, #6256, #2142)]
    |-- PyTorch SymmetricMemory (#202454)--> LLVM IR Bitcode Wrapper -> Triton rocSHMEM alltoall/broadcast
    |-- ROCm RCCL & ROCr (#14146/#14694)--> Zen 2/3/4 CPUID 23/25 Rome Restore (5 Consumers + GDR PHB)
    |-- Mooncake TENT & Store (#5026/#4787)-> XPU IPC Exporter pidfd Watch (0 Leak) + string_view LRU (-50%)
    +-- LMCache & NIXL (#6256/#2142/#2560)-> ZMQ NOBLOCK+POLLOUT + Overflow-Safe covers() + UCX Disconnect
```

---

## 3. Deterministic Offline Verification (Pillar 2 & Pillar 3)

Verified via [`experiments/megatron_thd_cudagraph_minf_snapshot_dsv4_flops_te_mxfp8_rocshmem_rccl_mooncake_nixl_phase181/run.py`](../../experiments/megatron_thd_cudagraph_minf_snapshot_dsv4_flops_te_mxfp8_rocshmem_rccl_mooncake_nixl_phase181/run.py) `--check`:
- `megatron_thd_cudagraph_gb200_checks_passed = 288`, `megatron_thd_cudagraph_pass_rate = 1.0`
- `megatron_minf_rl_interrupted_tokens_lost = 0`, `megatron_dsv4_hybrid_flops_cases_passed = 48`, `megatron_pp_retained_input_stream_races = 0`
- `te_batched_linear_mxfp8_supported = True`, `te_capturable_adam_redundant_unscaled_writes_at_unit_scale = 0`
- `pytorch_rocshmem_triton_device_collectives = 2`, `rccl_zen_rome_restored_consumers = 5`, `rocr_svm_pointer_type_id = 7`
- `mooncake_xpu_ipc_exporter_exit_leaked_vram_bytes = 0`, `mooncake_shard_allocator_key_copy_reduction_ratio = 0.50`
- `lmcache_zmq_blocked_client_head_of_line_stalls = 0`, `nixl_unbounded_file_covers_finite = 1`, `nixl_finite_covers_wrapping_range = 0`, `nixl_section_error_orphaned_ucx_connections = 0`

---

## 4. Enterprise Business Scenario Playbook (Pillar 4)

| Scenario Dimension | ✅ Good Fit | ⛔ Not For |
| :--- | :--- | :--- |
| **GB200 Sequence-Packed Pretraining & Interrupted RL Rollouts** | Enable `dp_balanced` THD sequence-packed TE attention CUDA graphs on GB200 (`#8361`, `288/288` checks) and call `MInf` `snapshot_live_requests()` before RL rollout preemption (`#8137`). | Reusing a captured THD attention graph when packed microbatch counts overflow the captured graph capacity (`#8361` explicitly rejects overflow). |
| **AMD Zen 2/3/4 + MI300/MI350 Collectives & In-Kernel rocSHMEM** | Deploy RCCL with `#14146` to restore Milan/Genoa `1H16P`/`4H4P` presets and same-root-complex GPU-NIC `GDR PHB`, and use PyTorch Triton `rocSHMEM` `alltoall`/`broadcast` (`#202454`). | Running unpatched NCCL 2.31-synced RCCL on Rome/Milan/Genoa topologies without `noCpuCheck` (loses `GDR PHB` due to CPUID `23/25` vs `143/175` mismatch). |
| **Cross-Process KV Cache Fabrics (Mooncake / LMCache / NIXL)** | Enable Mooncake `#5026` `pidfd` exporter monitoring, LMCache `#6256` `NOBLOCK+POLLOUT` ZMQ loop, and NIXL `#2142/#2560` overflow-safe file range checks and error disconnect. | Erasing `remoteBackends_[remote]` on `loadRemoteSections` failure without disconnecting backends (leaves stale UCX `remoteConnMap` entries that fail reconnects with `NIXL_ERR_INVALID_PARAM`). |
