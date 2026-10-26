# FlashAttention-4 SM90 Block-Sparse Tile Fit & SM100 In-Kernel TMA Padding, Megatron-LM Device-Initiated GroupedLinear Paged Stash & DSA FLOPs, Mooncake Target-Grouped Transfers / HA Checksum / Wildcard Rail / Lock-Free DFS Topology, NIXL Context Span & Dynamo Mocker Metrics Deep Dive (Phase 90)

> **中文版本**: [fa4-sparse-mla-tma-megatron-stash-dsa-flops-mooncake-nixl-dynamo-phase90.md](fa4-sparse-mla-tma-megatron-stash-dsa-flops-mooncake-nixl-dynamo-phase90.md)  
> **Evidence Tier**: `implementation` (first-hand commit inspection) + `own-offline` (deterministic Python 3.12 stdlib model in `experiments/fa4_sparse_mla_tma_megatron_stash_dsa_flops_mooncake_nixl_dynamo_phase90/run.py`). No live GPU/RDMA hardware benchmarks are claimed.

## 1. First-Hand Code Evidence & Implementation Paths

1. **FlashAttention-4 SM90 Block-Sparse Tile Fitting & SM100 Sparse MLA `1..128` Q-Head In-Kernel TMA Padding** (`Dao-AILab/flash-attention` PRs `#2903` `d9c4d15f5c`, `#2883` `d15f1531a4`):
   - `_fit_sm90_fwd_tile_to_block_sparsity` shrinks the SM90 forward heuristic tile (`[128, 80]` at `head_dim=192`) to `[128, 64]` so it evenly divides `sparse_block_size_kv=128` instead of raising `ValueError`;
   - SM100 Sparse MLA supports arbitrary `1..128` Q heads (including 64-head TP shards) via in-kernel TMA zero-padding into the 2-CTA UMMA 128-row tile, requiring $0\text{ B}$ of global memory padding copies.
2. **Megatron-LM Device-Initiated GroupedLinear Paged Stash & Exact DSA Top-K / Indexer FLOPs Accounting** (`NVIDIA/Megatron-LM` PRs `#6828` `f9d606dc47`, `#7043` `705274093a`):
   - Enables sync-free **Paged Stash** with device-initiated `GroupedLinear` and `hybridep` flex dispatcher;
   - Adds `dsa_flops.py` to compute causal Top-K effective KV tokens $\sum_{i=0}^{S-1}\min(i+1, K_{\text{top}})$ and Lightning Indexer FLOPs ($12.11\%$ of dense causal attention KV pairs at $S=32768, K_{\text{top}}=2048$).
3. **Mooncake Target-Grouped Transfers, HA Promotion Checksum Preservation, TENT Wildcard Direct Rail & Lock-Free DFS Shard Topology** (`kvcache-ai/Mooncake` PRs `#4799` `b46254dea9`, `#4889` `83f8f87749`, `#4923` `58e243fb29`, `#4770` `ad546de5e0`):
   - Groups multi-buffer `put/upsert/get` by target endpoint ($64\times$ fewer Transfer Engine batch submissions for 256 keys across 4 targets) and uses 6 async streams for Ascend local copies;
   - Persists `object_checksum` (`uint64_t`) in `StandbyObjectMetadata` and restores it in `MasterService::PublishRestoredObjects` across HA failover;
   - Preserves the configured direct peer rail when `RailMonitor::findBestRemoteDevice` evaluates wildcard remote NUMA targets;
   - Replaces `std::atomic_load_explicit(std::shared_ptr)` in `ShardAllocator` (which serialized all threads on a single `libstdc++` global mutex bucket) with lock-free immutable topology reads.
4. **NIXL `beginContextSpan` Exact Context Span ID & Dynamo Native Mocker Metrics Alignment** (`ai-dynamo/nixl` PR `#2508` `c8b46078e4`, `ai-dynamo/dynamo` PR `#17200` `59d7ef4e68`):
   - `beginContextSpan` starts a span whose ID equals the propagated trace context's `span_id`, unifying parent-child NVTX/Chakra trace trees across initiator and target;
   - Aligns Dynamo's native Rust mocker metrics and labels with vLLM, SGLang, and TRT-LLM.

## 2. Architect Exercises & Falsifiable Hypotheses

- **Sizing Exercise**: At $S=32768, K_{\text{top}}=2048$, causal DSA evaluates $65,012,736$ KV pairs vs. $536,887,296$ for dense causal attention ($12.11\%$).
- **Falsifiable Hypothesis**: Calling `std::atomic_load_explicit` on a shared `std::shared_ptr` inside `ShardAllocator::UpdateAccess` serializes concurrent `BatchGet` workers on a single `libstdc++` spinlock bucket.
