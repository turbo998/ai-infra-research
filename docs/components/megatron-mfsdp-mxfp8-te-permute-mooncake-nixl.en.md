# Megatron-LM MFSDPv2 Phase-Selective MXFP8 Plane All-Gather & EP A2A Segment Recompute, TransformerEngine Mask-Map Capacity Clamp, and Mooncake/NIXL/LMCache Low-Latency Fabrics

> **Bilingual Navigation / 双语导航**: [中文版](megatron-mfsdp-mxfp8-te-permute-mooncake-nixl.md) | [English Version](megatron-mfsdp-mxfp8-te-permute-mooncake-nixl.en.md)
> **Evidence IDs**: `src-megatron-mfsdp-mxfp8-te-permute-mooncake-nixl-2026` · `ev-megatron-mfsdp-mxfp8-te-permute-mooncake-nixl-2026` (`priority: 73`, Phase 64 Track C)
> **Reproducible Experiment**: `python experiments/megatron-mfsdp-mxfp8-te-permute-mooncake-nixl/run.py --check`

---

## 1. Architect Problem Statement & Engineering Context

Across Blackwell MXFP8 distributed training (Megatron-FSDPv2 and TransformerEngine), Expert-Parallel (EP) All-to-All activation recomputation, and RDMA/IPC KV cache and telemetry fabrics (Mooncake, NIXL, LMCache), systems architects encounter five critical scaling and memory-safety bottlenecks:

1. **Redundant Dual-Plane MXFP8 All-Gathers in MFSDPv2**: MXFP8 weights maintain separate `rowwise` (read by forward GEMMs) and `columnwise` (read by backward dgrad GEMMs) physical planes (FP8 data + E8M0 scales). Gathering all four planes on every unshard doubles cross-GPU All-Gather traffic and unsharded buffer footprint.
2. **State Drift in EP All-to-All Overlap Per-Segment Activation Recompute**: When combining `--overlap-moe-expert-parallel-comm` with `--recompute-granularity full`, layer forwards mutate `ModelChunkState` fields in place—including rolling `input_ids` and `mtp_input_mask` together for Multi-Token Prediction (MTP). Omitting `mtp_input_mask` from the segment snapshot causes backward replay to roll an already-rolled mask, detaching the wrong embedding rows.
3. **Out-of-Bounds Writes in TransformerEngine Triton Mask-Map `te_permute`**: When `num_out_tokens < routing_map.sum()` under capacity dropping, `_row_id_map_pass_2_kernel` left `num_out_tokens` unused and wrote over-capacity rows past the end of the output buffer.
4. **Per-WR Clock Reads Under Lock in Mooncake RDMA `submitPostSend` & Metadata Shard Contention**: Calling `clock_gettime(CLOCK_REALTIME)` (~`19 ns/WR`) for every slice inside a 256-WR post chain while holding `lock_` dominates chain construction and delays the NIC doorbell; meanwhile, running DFS eviction and soft-pin expiry across 1,024 metadata shard locks stalls online KV metadata access.
5. **False Sharing in NIXL `sharedRingBuffer` and Next-Step Polling Bubble in LMCache Lookups**: Placing `write_pos` and `read_pos` on the same cache line and reloading the remote atomic index on every `push()`/`pop()` causes constant cache-line bouncing; in LMCache, waiting for the next vLLM scheduler step after a `LOOKUP` ack before sending `QUERY_PREFETCH_STATUS` adds a full step bubble.

---

## 2. Core Source Changes & Mathematical Modeling

### 2.1 Megatron-LM MFSDPv2 Phase-Selective MXFP8 Plane All-Gather (`#7907`, `18bc49d`) & EP A2A Segment Recompute (`#6429`, `da2c0ec`)

1. **Phase-Selective MXFP8 All-Gather (`Phase.FORWARD` vs `Phase.BACKWARD`)**:
   - `FsdpModule` replaces `_unshard_event` with `_unshard_events: dict[Phase, torch.cuda.Event]`.
   - `QuantizedDBuffer.redistribute(..., rowwise=rowwise, columnwise=columnwise)` gathers **only `rowwise_data` + `rowwise_scale_inv` during `Phase.FORWARD`** and **only `columnwise_data` + `columnwise_scale_inv` during `Phase.BACKWARD`**.
   - `update_mxfp8_tensor` pads only the newly gathered phase's scales while preserving the other phase's installed planes and readiness event.
   - **Bandwidth Savings**: For a $[14336, 5376]$ MXFP8 weight ($B=32$), a single plane (data + scale) is **`75.7969 MiB`** vs **`151.5938 MiB`** for both planes. Across a forward + resharded-backward step, total All-Gather volume drops from **`303.1875 MiB` to `151.5938 MiB` (-50.0%)**.
2. **Per-Segment EP A2A Overlap Full Recompute (`#6429`, `da2c0ec`)**:
   - `RecomputeSegment` runs initial segment forwards under `torch.no_grad()`, strips accidental `grad_fn` attachments via `.detach()`, and calls `reset_for_recompute()` immediately after each segment while keeping tensor storages intact for replay.
   - Snapshots all 5 mutable chunk-state fields `_MUTABLE_CHUNK_STATE_FIELDS = ("input_ids", "position_ids", "padding_mask", "mtp_input_mask", "mtp_hidden_states")`, preventing double-rolling of `mtp_input_mask` during backward replay.

### 2.2 TransformerEngine Triton Mask-Map Capacity Drop (`#3347`, `c9c9225`) & `GroupedTensor` Autograd/Stream Tracking (`#3701`, `2199b71`)

1. **Triton `_row_id_map_pass_2_kernel` Capacity Clamp**: Passes `num_out_tokens` into `_row_id_map_pass_2_kernel` and clamps `row_id = tl.where(row_id >= num_out_tokens, -1, row_id)` (while JAX passes `2**31 - 1`). On a `4096 tokens x 8 experts` routing map (`16,420` routed pairs, capacity `16,403`), it keeps exactly `16,403` rows and marks the **17 excess entries as `-1`**, preventing 17 out-of-bounds row writes.
2. **`GroupedTensor` Identity Views & `record_stream` (`#3701`, `2199b71`)**: `make_wrapper_like` sets `requires_grad=False` so autograd attaches view metadata cleanly after `__torch_dispatch__`, and `aten.record_stream` forwards stream tracking across all **7 backing tensors** (`rowwise_data`, `columnwise_data`, `scale_inv`, `columnwise_scale_inv`, `amax`, `first_dims`, `tensor_offsets`).

### 2.3 Mooncake Single-Clock RDMA Post Chain (`#4793`, `dd1217b`) & Shard-Free Eviction (`#4854`, `c81f4ad`), NIXL Ring Buffer (`#2018`, `a9f1f7a`), and LMCache Callback LookupState (`#5960`, `ff9185d`)

1. **Mooncake `RdmaEndPoint::submitPostSend` Clock Read Once Per Chain (`#4793`)**: Hoists `const int64_t post_ts = getCurrentTimeInNano()` outside the per-WR loop so all WRs in an `ibv_post_send` chain share one timestamp. On a 256-WR call (two 128-WR chains, `author-benchmark`), lock hold and doorbell setup latency drops from **`6.3 us` to `1.9 us` (`25.0 ns -> 7.5 ns` per WR, `3.3158x` speedup)**.
2. **Mooncake Store Shard-Free DFS Eviction & `DeferDeletion` (`#4854`)**: Moves DFS eviction, soft-pin expiry, and in-flight tracking off the 1,024 metadata shards using `ReadCursor(tenant_id, CursorSlice)` and adds `ImmutableBucketAllocator::DeferDeletion(bucket, not_before)` so evicted bucket files remain on disk until active reader leases expire.
3. **NIXL `sharedRingBuffer` 256-Byte Alignment & Cached Positions (`#2018`, `a9f1f7a`)**: Bumps `TELEMETRY_VERSION = 6`, separates `write_pos` and `read_pos` with `alignas(256)` (`DESTRUCTIVE_INTERFERENCE_SIZE = 256`, keeping `version` fixed at byte offset 16), and caches `cachedReadPos_` / `cachedWritePos_` so `push()` only reloads `read_pos` when the ring appears full—cutting cross-core acquire loads by **99.9%** (`1000 -> 1` over 1,000 pushes on a 1,024-slot ring).
4. **LMCache Callback-Driven `LookupState` (`#5960`, `ff9185d`)**: Adds `MessagingFuture.add_done_callback` so each server's `LOOKUP` ack immediately triggers `QUERY_PREFETCH_STATUS` on the transport receive thread without waiting for the next scheduler step (reducing lookup-to-query resolution from `5.40 ms` to `0.80 ms`, a **`6.75x`** speedup at a `5.0 ms` step interval).

---

## 3. Deterministic Verification Summary (`own-offline` + `author-benchmark`)

Running `python experiments/megatron-mfsdp-mxfp8-te-permute-mooncake-nixl/run.py --check` verifies:

| Subsystem / Repo | Optimization | Baseline / Before Fix | Current Implementation | Key Architectural Gain |
|---|---|---|---|---|
| **Megatron-LM MFSDPv2** (`#7907`) | Phase-selective MXFP8 All-Gather | `303.19 MiB/step` | **`151.59 MiB/step`** | **-50.0% All-Gather communication volume** |
| **Megatron-LM EP A2A** (`#6429`) | `RecomputeSegment` + 5-field snapshot | Missing `mtp_input_mask` double-rolls | **Snapshots all 5 mutable fields + `no_grad`** | Prevents silent MTP gradient corruption |
| **TransformerEngine** (`#3347`, `#3701`) | Triton mask-map clamp + `record_stream` | 17 OOB row writes; untracked views | **`row_id >= num_out_tokens -> -1`; 7 buffers** | Eliminates permute OOB & multi-stream UAF |
| **Mooncake RDMA** (`#4793`) | Single clock read per RDMA post chain | `6.3 us` (`25.0 ns/WR`, 256 calls) | **`1.9 us` (`7.5 ns/WR`, 2 calls)** | **`3.32x` faster lock hold & doorbell** |
| **NIXL & LMCache** (`#2018`, `#5960`) | `alignas(256)` cached pos + future callbacks | Per-push acquire load; next-step poll | **`-99.9%` cross-core loads; `5.4 -> 0.8 ms`** | Eliminates false sharing & step polling bubble |
