# Megatron-LM Hybrid DSA-GQA & GDN CUDA-Graph Prefill, Mooncake Store Cross-Topology KV Resharding & Tenant Quota, and NIXL Telemetry

> **Upstream Source Commits (2026-10-20 ~ 2026-10-21)**:
> - `NVIDIA/Megatron-LM` PR `#7156` (`d14b699755061d55cb918f68500fba24401a1846`): `feat: Support DSA over GQA in hybrid models and fix DSA indexer loss logging (#7156)`
> - `NVIDIA/Megatron-LM` PR `#8353` (`29e1a7b6697c61d184c95726865d417fc51b8574`): `feat(ssm): support CUDA graph replay for GDN dynamic inference prefill via single-update GDP (#8353)`
> - `kvcache-ai/Mooncake` PR `#4762` (`e2d6e8e0f2df113d84f9e39379f98c4b67b199ab`): `feat(reshard): add Store KV cache reuse geometry planner (#4762)`
> - `kvcache-ai/Mooncake` PR `#4809` (`44c4d9cf4ace3ce2955312a2467b1305d910dd48`): `feat(store): integrate TenantQuotaPolicy with upsert carry-over and watermark eviction (#4809)`
> - `ai-dynamo/nixl` PR `#2479` (`03d9b74648cc06daef8aa71f15a53c5a98be1319`): `feat(telemetry): add nixl.xfer_request_id and nixl.xfer_execution counters (#2479)`

---

## 1. Architectural Problem Statement

Across hybrid SSM-Attention model execution (Megatron-LM) and disaggregated multi-tenant KV cache storage (Mooncake Store + NIXL), three cross-layer challenges arise:

1. **Missing DSA-over-GQA Support & Non-Graph-Replayable GDN Varlen Prefill in Megatron-LM**:
   - DeepSeek Sparse Attention (DSA) was previously coupled exclusively to MLA and could not be attached to GQA layers in hybrid Mamba/GDN architectures, while `DSAIndexerLossLoggingHelper` averaged already-scaled losses, distorting reported metrics by `loss_scale`;
   - Gated DeltaNet (`gdn.py`) dynamic inference prefill previously called `causal_conv1d` and `chunk_gated_delta_rule`, whose workspace allocations and launch grids depended on runtime sequence lengths, breaking CUDA Graph replay.
2. **Redundant Replica Uploads in Cross-Topology KV Resharding & Upsert Quota Undercharging in Mooncake Store**:
   - When a prefill cluster holds replicated KV heads (`replica_ordinal > 0`), uploading from every participant writes duplicate KV shards; when a decode cluster with a different TP/PP topology loads those shards, Cartesian fragment scans incur quadratic planning overhead;
   - During an in-place key upsert in a multi-tenant memory pool, the replaced object's memory remains pinned until the new `PROCESSING` memory replica settles. Dropping the old charge immediately upon replacement lets concurrent upserts transiently exceed physical pool memory.
3. **Ambiguous Multi-Repost Request Lineage in NIXL Telemetry**: Because a single `nixlXferReqH` handle can be reposted multiple times via `postXferReq`, operators lacked a non-zero 64-bit request identifier paired with an atomic execution counter incremented strictly on validated backend submissions.

---

## 2. Mathematical Formulation & System Design

```
+--------------------------------------------------------------------------------------------------+
|  Megatron DSA-GQA + GDN Graph (#7156/#8353) | Mooncake Reshard + Quota (#4762/#4809) | NIXL      |
+--------------------------------------------------------------------------------------------------+
|                                                                                                  |
|  [1] Megatron-LM Hybrid DSA-GQA & GDN Single-Update GDP CUDA-Graph Prefill                       |
|      * DSGroupedSelfAttention + SimplifiedDSGQAIndexer + raw_values unscaled loss logging        |
|      * GDN = GDP with num_householder = 1: preallocated metadata.gdp_chunk_indices / offsets     |
|      * causal_conv1d_varlen_fn + chunk_gated_delta_product_varlen write state in place           |
|                                                                                                  |
|  [2] Mooncake Store KV Reshard Planner (#4762) & TenantQuotaPolicy (#4809)                       |
|      * Upload: filter `part.replica_ordinal == 0` -> 1 writer per unique shard                   |
|      * Load: `replicas=True` + `HeadIntervalIndex.cover(head_start, end)` -> fill all replicas   |
|      * Quota words: Held = policy_words[0], Carried = policy_words[1] (kept while PROCESSING)    |
|      * Watermark eviction: evict (used_ratio - (watermark - eviction_ratio)) * effective_quota   |
|                                                                                                  |
|  [3] NIXL Transfer Telemetry (#2479)                                                             |
|      * 64-bit `nixl.xfer_request_id` (redraws if PRNG yields 0) + atomic `nixl.xfer_execution`   |
+--------------------------------------------------------------------------------------------------+
```

### 2.1 Megatron-LM Hybrid DSA-GQA & GDN Single-Update GDP CUDA-Graph Prefill (PR `#7156` / PR `#8353`)

- **Hybrid DSA over GQA & Unscaled Loss Tracking (PR `#7156`)**: Introduces `DSGroupedSelfAttention`, `DSGQACoreAttention`, and `SimplifiedDSGQAIndexer` with `_UNFUSED_DSA_KERNEL_BACKENDS = ("none", "min-memory-triton", "min-memory-torch", "reference")`, and records unscaled `raw_values` in `DSAIndexerLossLoggingHelper` alongside backward-scaled losses.
- **GDN Varlen Prefill as `num_householder = 1` GDP for CUDA Graph Replay (PR `#8353`)**: Recognizes that Gated DeltaNet is Gated DeltaProduct with `num_householder = 1`. `ssm_prefill` switches to `causal_conv1d_varlen_fn` (consuming preallocated `conv_seq_idx` and `conv_seq_start`) and `chunk_gated_delta_product_varlen` (consuming fixed-capacity `gdp_chunk_indices`, `gdp_chunk_indices_dp`, and `gdp_chunk_offsets` refreshed in place before replay), with FP32 gates $g = -\exp(A_{\log})\cdot\operatorname{softplus}(\alpha + \text{dt\_bias})$ and $\beta = \sigma(\beta_{\text{raw}})$.

### 2.2 Mooncake Store KV Reuse Geometry Planner & TenantQuotaPolicy (PR `#4762` / PR `#4809`)

- **Replica-Deduplicated Upload & Interval-Indexed Load Planning (PR `#4762`)**:
  - `plan_kv_cache_store_upload` filters `part.replica_ordinal == 0` (`replicas=False`) so each unique `(layer_ids, head_start, head_count)` slice is uploaded by exactly one writer;
  - `plan_kv_cache_store_load` selects all target replicas (`replicas=True`) and queries per-`(layer, component)` `HeadIntervalIndex.cover(part.head_start, end)` to slice stored fragments with exact relative byte offsets $\text{part\_offset}' = \text{part\_offset} + (\text{start} - \text{frag\_start}) \cdot \text{head\_stride\_bytes}$, bounded by `max_parts = 100,000`, `max_fragments = 100,000`, and `max_bytes = 1 TiB`.
- **Two-Word Upsert Quota Accounting & Slab Watermark Eviction (PR `#4809`)**:
  `TenantQuotaPolicy` stores `Held = policy_words[0]` and `Carried = policy_words[1]` per key. On `OnReplace`, the replaced object's charge is added to `Carried` and retained until `HasProcessingMemoryReplica(metadata)` becomes `False` in `OnWriteRelease`:
  $$\text{owed} = \operatorname{SaturatingAdd}(\operatorname{MemoryCharge}(\text{metadata}), \text{Carried})$$
  When `used_ratio > watermark`, `OnMaintenance` evicts down to `target_ratio = max(0.0, watermark - eviction_ratio)` in one slab (`target_bytes = (used_ratio - target_ratio) * effective_quota_bytes`).

### 2.3 NIXL Non-Zero `xfer_request_id` & `xfer_execution` Telemetry (PR `#2479`)

NIXL assigns a 64-bit random `nixl.xfer_request_id` (redrawing if the PRNG produces `0`) when tracing is enabled, and increments an atomic `nixl.xfer_execution` counter only after repost/notification validation succeeds immediately before `backend->postXfer`, propagating both fields into `nixl::xfer.complete`.

---

## 3. Deterministic Verification & Benchmark Summary

Running `python experiments\megatron_dsa_gqa_gdn_mooncake_store_reshard_quota\run.py --check` verifies all invariants:

| Module | Scenario | Baseline / Buggy Behavior | New Mechanism | Verified Result |
|---|---|---|---|---|
| **Megatron GDN CUDA Graph (`#8353`)** | 3 varlen sequences `cu_seqlens=[0,110,174,310]`, `chunk_size=64`, cap `12` | Dynamic grid breaks CUDA graph replay | **`6` active chunks + `6` `(-1,-1)` pad slots, `num_householder=1`** | **100% static launch grid & workspace** |
| **Megatron DSA Loss Logging (`#7156`)** | 4 layers `[0.42,0.38,0.35,0.45]`, `loss_scale=0.1` | Logs scaled mean `0.04` (`10x` distortion) | **`raw_values` preserves true mean `0.40`** | **Decouples telemetry from backward scale** |
| **Mooncake Store Reshard (`#4762`)** | Source `TP=2` (2 replicas, 4 parts) $\to$ Target `TP=4` (2 replicas, 8 parts) | 4 duplicate uploads (`2x` write amp) | **`2` unique upload writers, `32` interval-covered load ranges** | **`-50.0%` upload bytes, full target replica fill** |
| **Mooncake Tenant Quota (`#4809`)** | Quota `1000B`, `400B` replaced by `900B` upsert (`watermark=0.85, evict=0.15`) | Undercharges in-flight upsert / thrash evicts | **Charges `1300B` while `PROCESSING`, settles to `900B`, evicts `200B` slab** | **Zero upsert memory overcommit** |
| **NIXL Telemetry (`#2479`)** | 2 zero PRNG draws + 4 calls (2 failed validation) | `id=0` ambiguity & inflated attempt count | **`2` zero redraws, `xfer_execution = 2` on actual posts** | **Exact request & execution lineage** |

---

## 4. Production Guidance & Boundaries

- **When to use**: Hybrid GDN + DSA-GQA training and graph-captured prefill in Megatron-LM, cross-topology prefix KV reuse and multi-tenant memory isolation in Mooncake Store, and multi-repost RDMA telemetry in NIXL.
- **Boundaries**: Megatron GDN graph-safe prefill requires total active chunks across the batch to fit within the preallocated `gdp_chunk_indices` capacity and does not support chunked prefill (`not context.is_chunked_prefill_enabled()`).
