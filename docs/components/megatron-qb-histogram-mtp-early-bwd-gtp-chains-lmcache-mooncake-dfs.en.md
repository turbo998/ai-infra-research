# Deep Dive: Megatron-LM In-Place Vocab-Parallel Cross-Entropy & Early-Backward MTP Loss, Kimi-K3 Global-Batch Quantile Balancing, Split Graphed MoE GTP Chains, LMCache Refcount Guard, and Mooncake Sliced Tenant Cursors

- **Research Date**: 2026-10-24 (Phase 58 · Priority 67 · Track C)
- **Upstream Evidence Chain**:
  - `NVIDIA/Megatron-LM@6e5946498f` (`#7478`, *perf(ce,mtp): keep VocabParallelCrossEntropy in-place under torch.compile and add mtp_loss_early_backward*)
  - `NVIDIA/Megatron-LM@92641a421f` (`#7394`, *feat(moe): Kimi K3 global-batch quantile balancing and CUDA graph capture cleanup*)
  - `NVIDIA/Megatron-LM@9e5aad0c6a` (`#8345`, *fix(gtp): split graphed MoE expert FC1/FC2 prefetch chains*)
  - `NVIDIA/Megatron-LM@85bf0a667c` (`#8344`, *refactor(ssm): share Mamba branch execution via local ResidualBranchInput*)
  - `NVIDIA/Megatron-LM@3630e44a80` (`#7736`, *fix(dist): forward timeout to DYNAMIC_DP_CP_GROUP*)
  - `LMCache/LMCache@562517ad9f` (`#5940`, *fix(storage): release memory_objs in batched_put try/finally*)
  - `LMCache/LMCache@48fc9d4849` (`#5696`, *fix(vllm): support participates_in_prefix_caching fallback in is_scratch_spec*)
  - `kvcache-ai/Mooncake@c81f4ad486` (`#4854`, *refactor(master): move DFS eviction and soft-pin off metadata shards onto sliced ReadCursor and DeferDeletion*)
  - `kvcache-ai/Mooncake@e764353f4e` (`#4858`, *feat(master): add HTTP /api/v1/soft_pin and /api/v1/soft_unpin and raise max TTL to 7d*)
  - `kvcache-ai/Mooncake@5872ee870d` (`#4844`, *fix(store): bound processing replica cleanup via tenant iteration*)
- **Evidence Tier**: `implementation` + `own-offline` (`experiments/megatron-qb-histogram-mtp-early-bwd-gtp-chains-lmcache-mooncake-dfs/run.py`)

---

## 1. Architectural Motivation & Core Bottlenecks

Across large-scale MoE pretraining and distributed KV storage planes, Phase 58 Track C resolves five critical bottlenecks:

1. **Transient `4 GiB` Allocation in Compiled `VocabParallelCrossEntropy` & Multi-Head `[S, B, V]` Softmax Accumulation in MTP (`Megatron-LM #7478`)**:
   - Under `torch.compile` (`Inductor`), combining an in-place `[S, B, V]` mutation with a reduction (`exp_logits.sum`) or scatter (`grad[idx] -= ...`) in the same compiled graph forces Inductor to materialize a fresh `[S, B, V]` buffer and copy it back (`4.0 GiB` extra transient memory and `~2x` runtime for an `8K x 128K` FP32 head).
   - Furthermore, Multi-Token Prediction (`mtp_num_layers = K`) previously kept the `[S, B, V]` FP32 softmax of all `K` MTP heads alive until the main backward pass (`+4.0 GiB` per MTP head).
2. **Kimi-K3 Global-Batch Quantile Balancing Histogram Pollution on Dropless Reruns & Failed CUDA Graph Capture (`Megatron-LM #7394`)**: When a dropless MoE step overflows and triggers `prepare_for_rerun()`, unreset `qb_histogram` buffers double-count tokens from the failed attempt; meanwhile, an exception inside `create_cudagraphs()` could leave capture flags active and disable DDP backward hooks permanently.
3. **Graphed MoE Expert FC1/FC2 Prefetch Chain Collision in GTP (`Megatron-LM #8345`)**: Placing captured MoE FC1 and FC2 weights into the single `GTP_graphed` chain broke one-block-ahead grouped-expert prefetch ordering.
4. **Pinned Memory Leaks in LMCache `StorageManager.batched_put` on Backend Exceptions (`LMCache #5940`)**: Without a `try...finally` block, an exception in any storage backend skipped `ref_count_down()` on all allocated `memory_objs`.
5. **Mooncake Metadata Shard Lock Contention & Premature DFS Bucket Deletion (`Mooncake #4854, #4858`)**: Scanning all `1,024` metadata shards for DFS eviction and soft-pin expiry contended with foreground traffic, and deleting an evicted bucket file before active reader leases expired risked use-after-delete errors.

---

## 2. Mechanism Deep Dive & Code Walkthrough

### 2.1 Megatron-LM Split In-Place `VocabParallelCrossEntropy` & `mtp_loss_early_backward` (`#7478`)

- **Pointwise-Only `jit_fuser` Graphs (`fused_cross_entropy.py`)**: Splitting `_gather_predicted_logits` (read-only), `_exp_logits_inplace` (pointwise), and `_scale_gradients_inplace` (pointwise) while running reductions and target scatters outside the compiled graphs keeps Inductor mutations strictly in place, saving **`4.0 GiB`** of transient memory on `S*B = 8192, V_local = 131072` FP32.
- **`mtp_loss_early_backward=True` (`multi_token_prediction.py`)**: Runs `fwd(head_k) -> bwd(head_k)` immediately inside `process_mtp_loss` on a detached `head_hidden`, freeing each MTP head's `4.0 GiB` `[S, B, V]` softmax before head `k+1` starts and injecting only the `64 MiB` `[S, B, H]` BF16 gradient via `_MTPHeadGradInjector`. For `K = 3` MTP heads, live head memory drops from **`16.0 GiB` to `4.1875 GiB` (`-73.83%`)** with zero recomputation.

### 2.2 Kimi-K3 Global-Batch Quantile Balancing (`#7394`) & Split Graphed MoE GTP Chains (`#8345`)

- **Kimi-K3 Histogram Pooling & Rerun/Capture Cleanup (`#7394`)**: Registers a `[num_experts, moe_router_qb_num_bins=1000]` `int32` histogram (`1,000 KiB`/layer for `E=256`), zeroes it in `prepare_for_rerun()` via `_reset_qb_histograms()` while preserving CUDA addresses, and wraps `create_cudagraphs()` in `try...finally` so failed capture resets `_set_capture_end()` and `te_set_capture_end()`.
- **Split Graphed MoE Chains (`#8345`)**: `get_graphed_chain_ids()` returns `("GTP_graphed", "GTP_remat_grouped_fc1_graphed", "GTP_remat_grouped_fc2_graphed")`, isolating FC1 and FC2 traversal lists while sharing a single IB stream key.

### 2.3 LMCache `batched_put` `try...finally` (`#5940`) & Mooncake `CursorSlice` + `DeferDeletion` (`#4854, #4858`)

- **LMCache (`#5940`, `#5696`)**: Wraps `backend.batched_put` loops in `try...finally` to guarantee `ref_count_down()` on all chunks (`0` leaked buffers), and adds `participates_in_prefix_caching=False` fallback in `is_scratch_spec`.
- **Mooncake (`#4854`, `#4858`)**: Introduces `ReadCursor(tenant_id, CursorSlice{index, count})` for lock-friendly parallel tenant walks, adds `ImmutableBucketAllocator::DeferDeletion(bucket, not_before)` so evicted DFS bucket files remain on disk until active lease deadlines pass, and exposes `POST /api/v1/soft_pin` and `/api/v1/soft_unpin` (`1..4096` keys, raising `max_kv_soft_pin_ttl` from `24h` to **`7d` / `604,800,000 ms`**).

---

## 3. Architect Exercises & Falsifiable Questions

1. **Quantitative Exercise**: Compute the per-layer memory of `qb_histogram` for `E = 256, num_bins = 1000` in `int32` (`1,000 KiB`) and compare the all-reduce payload against gathering raw per-token router scores across `65,536` global batch tokens (`65,536 * 256 * 4 B = 64 MiB`, a **`65.5x`** communication reduction).
2. **Falsifiable Hypothesis**: Partitioning `N = 200` tenant objects across `CursorSlice{index, count}` for `index in 0..count-1` visits every object exactly once (`sum(len(slice_i)) == 200` and `len(union) == 200`).
