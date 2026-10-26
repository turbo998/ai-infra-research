# FlashAttention-4 SM90 Unmasked Interior Fast-Path & SM100 64-Head Sparse MLA Recompute-P, verl vLLM >= 0.28 Multi-Node Hybrid-LB Leader Guard, Mooncake Tiled COO Partial Reads / HA SIGTERM Lease Release / SSE4.2 CRC-32C, NIXL EP Barrier Put Completion / EFA Unsolicited Write Recv Disable, and Dynamo KV-Router Custom Visitor / OnceLock CRTC Leaf Storage Deep Dive (Phase 89)

> **中文版**: [fa4-sparse-mla-verl-mooncake-reshard-nixl-ep-dynamo-kvrouter-phase89.md](fa4-sparse-mla-verl-mooncake-reshard-nixl-ep-dynamo-kvrouter-phase89.md)  
> **Evidence Class**: `implementation` (pinned commit source review) + `author-benchmark` (upstream PR measurements, not re-run on local hardware) + `own-offline` (`experiments/fa4_sparse_mla_verl_mooncake_reshard_nixl_ep_dynamo_kvrouter_phase89/run.py` deterministic stdlib model)

---

## 1. Architectural Context & Problem Statement

In multi-node RL post-training (GRPO) and disaggregated serving clusters, end-to-end throughput and availability are bounded by **backward activation memory peaks on long sequences, multi-node rollout load-balancer protocol invariants, topology-independent sparse weight resharding, HA master lease handoff latency during graceful drains, one-sided RDMA/NVLink barrier source-buffer reuse races, and KV-router control-plane deserialization and radix-tree teardown overhead**. This document analyzes 10 non-overlapping commits across `Dao-AILab/flash-attention`, `volcengine/verl`, `kvcache-ai/Mooncake`, `ai-dynamo/nixl`, and `ai-dynamo/dynamo`.

---

## 2. Core Implementation & Distributed Invariant Analysis

### 2.1 FlashAttention-4: SM90 Interior Unmasked KV Fast-Path & SM100 64-Head Sparse MLA Recompute-P (`flash-attention@eed1971f51` #2904, `flash-attention@6a3ecb9be2`)
1. **Skipping Causal/Local Masking on SM90 Unmasked Interior Blocks (`flash-attention@eed1971f51`, PR #2904)**:
   - In `flash_attn/cute/flash_fwd_sm90.py`, interior KV blocks that require no causal/local masking previously still invoked `mask_fn`, executing **64 redundant `FSEL` instructions per iteration**.
   - Aligning SM90 with SM100 (passing `mask_fn` on interior blocks only when a custom `mask_mod` is present) leaves forward outputs and LSE bitwise unchanged while boosting H200 BF16 causal forward throughput from `613 -> 660 TFLOPS` (**`+7.67%`**) at `hdim128, 16k` and `463 -> 531 TFLOPS` (**`+14.69%`**) at `hdim64, 16k`.
2. **Allowing 64 Q Heads in SM100 Sparse MLA Recompute-P (`flash-attention@6a3ecb9be2`)**:
   - Relaxes `gather_bwd_recompute_p` from `num_head == 128` to `num_head in (64, 128)` since 64 Q heads fill an unpadded `tile_m = 64` backward tile (`pad_qheads = False`) while reusing `tiled_mma_VdO`.
   - **Author Benchmark (`author-benchmark`, GB200 64-head 64k tokens, `B=1, gather_width=2048, token_chunk=4096`)**: Trades `144.3 -> 150.8 ms` (`+4.5%` step time) for a **66.75% peak memory reduction (`41.2 -> 13.7 GiB`)** and cuts saved activations from `20.3 GiB` to `8.0 GiB`.

### 2.2 verl: Multi-Node vLLM >= 0.28 Hybrid-LB Leader Guard (`verl@3dd97a0e1b`, PR #8234)
- **Failure Mechanism**: Passing `data_parallel_start_rank` to the non-headless `node_rank == 0` API server causes vLLM $\ge 0.28$ (`EngineArgs.create_engine_config`) to infer `hybrid-LB` mode and reject all `--headless` remote follower engines (`Remote engine 15 must not use --headless in external or hybrid dp lb mode`).
- **Fix & Validation (`author-benchmark`)**: `vllm_async_server.py` now sets `data_parallel_start_rank` only when `self.node_rank > 0`. Validated on a **128-node GB200 cluster (16 replicas × 32 GPUs, DeepSeek-V4-Pro)** with end-to-end GRPO rollout/actor log-prob Pearson correlation of **`0.995`**.

### 2.3 Mooncake: Target-Independent Tiled COO Partial Reads, HA SIGTERM Lease Release & SSE4.2 CRC-32C (PR #3751, #4814, #4821)
1. **Target-Independent Tiled COO Sparse Reshard Planner (`Mooncake@880bcf6763`, PR #3751)**:
   - Adds `TiledCOOLayout` and `plan_tiled_coo_read` in `mooncake-reshard/python/mooncake/reshard/weight/tiled_coo.py`, indexing sparse coordinate/value deltas in globally aligned row-major tiles with per-record tile directories so arbitrary target TP/EP boxes can read partial intersections without pre-partitioning payloads for a fixed target topology.
2. **HA Leader Lease Release After SIGTERM Drain (`Mooncake@ac6b494e5b`, PR #4814)**:
   - Stops `StopHttpMetadataCleanup()` and `LeadershipRenewalGuard` after admin drain and explicitly invokes `ReleaseLeadership` before process exit (exiting with `kHaShutdownReleaseFailedStatus = 2` if unconfirmed), eliminating etcd lease TTL wait during planned rolling restarts.
3. **Runtime SSE4.2 Hardware `crc32` & Slice-by-8 CRC-32C (`Mooncake@93bc4ff541`, PR #4821)**:
   - Accelerates record checksums (`0x82F63B78` polynomial) in Offset Allocator and HA OpLog snapshots via runtime CPUID-detected SSE4.2 `crc32` instructions on x86-64 and 8-table Slice-by-8 elsewhere.

### 2.4 NIXL & Dynamo: EP Barrier Put Completion Wait, Libfabric EFA Unsolicited Write Recv Disable & KV-Router Optimization (PR #2442, #2494, #17238, #17259)
1. **NIXL EP Low-Latency Barrier Waits for Local Put (`nixl@48d531f9e6`, PR #2442)**:
   - Passes `nixlGpuXferStatusH` to `nixlPut` in `nixl_ep_ll.cu` and polls `nixlGpuGetXferStatus` before returning so the next barrier cannot decrement `sync_count[dst]` to `-2` before a CPU-proxied Put reads `-1`. Turns **0/5 into 5/5 passes** on GB200 NVLink + CPU proxy with Kineto profiling.
2. **Hard-Coded EFA Unsolicited Write Recv Disable (`nixl@410b8dc22e`, PR #2494)**:
   - Defines `NIXL_OPT_EFA_USE_UNSOLICITED_WRITE_RECV = -FI_PROV_SPECIFIC_EFA + 8` in `libfabric_rail.cpp` so wheels built against Libfabric 1.21 headers still call `fi_setopt()` on Libfabric $\ge 2.5$ runtimes, preventing silent loss of remote RDMA write completions under load.
3. **Dynamo KV-Router Visitor Deserialization & Lazy `OnceLock<ArcSwap>` CRTC Leaves (`dynamo@2d17836993` #17238, `dynamo@d1d61fc619` #17259)**:
   - Replaces `#[serde(flatten)]` and `serde_path_to_error` with direct `Visitor` parsing in `selection/types.rs` and wraps `NodeChildren.state` in `OnceLock<ArcSwap<ChildrenState>>` (`children.rs`) so childless CRTC leaf nodes never walk every thread's `arc_swap` debt list upon eviction.

---

## 3. Cross-Architecture Comparison (Fireworks / Together / Cerebras / Groq)

| Dimension | FA4 + verl + Mooncake + NIXL + Dynamo (Phase 89) | Fireworks AI / Together AI | Cerebras WSE-3 / Groq LPU |
| :--- | :--- | :--- | :--- |
| **Long-Context Activation Memory** | FA4 SM100 64-head Sparse MLA `recompute-P` cuts peak memory by **66.75%** (`41.2 -> 13.7 GiB`) for `+4.5%` step time | Essential for long-context fine-tuning and RL post-training density | Fixed on-wafer/on-chip SRAM requires streaming activation offload for 64k+ contexts |
| **One-Sided RDMA & KV Routing** | NIXL dual-sided barrier completion + EFA completion hardening + lock-free CRTC leaf eviction | Disaggregated prefill/decode pools depend on identical RDMA completion and prefix-routing guarantees | Deterministic intra-rack fabrics avoid dynamic KV prefix-tree eviction churn |

---

## 4. Architect Exercises & Falsifiable Questions

1. **One-Sided Put Source-Buffer Lifetime**: Trace how an asynchronous `nixlPut` reading `sync_count[dst]` from GPU global memory via a CPU proxy thread can observe `-2` instead of `-1` when `barrier()` only waits on `ld_acquire_sys_global` of the peer's incoming counter.
2. **Falsifiable Follow-Up**: Benchmark CRTC leaf eviction throughput in `dynamo_kv_router` across 32 worker threads before and after `OnceLock<ArcSwap<ChildrenState>>` when 80% of evicted prefix-tree nodes are childless leaves.
