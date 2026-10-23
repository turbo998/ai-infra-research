# Deep Dive: Megatron-LM MFSDPv2 Phase-Specific MXFP8 Plane Gathering, TorchTitan GB300 128-GPU RL Topology & Flux Latent Schedule, LMCache Callback Lookup, Mooncake Single Clock Read, and NIXL/Dynamo Atomic Isolation

> **Language / 语言**: [English](megatron-mfsdp-mxfp8-planes-lmcache-lookup-nixl-ring.en.md) | [中文](megatron-mfsdp-mxfp8-planes-lmcache-lookup-nixl-ring.md)
> **Question ID**: `megatron-mfsdp-mxfp8-planes-lmcache-lookup-nixl-ring` (`priority: 76`, Phase 67 Track C)
> **Evidence Class**: `implementation` + `author-benchmark` + `own-offline`

---

## 1. Problem Statement & Architectural Context

Across large-scale distributed training (MXFP8 FSDP and 816B RL training) and disaggregated KV cache / telemetry data planes, five bottlenecks directly govern communication volume, memory safety, and tail latency:

1. **Redundant Dual-Plane MXFP8 All-Gather in Megatron-LM MFSDPv2 & TE Capacity Overrun**:
   - MXFP8 weights maintain both rowwise (`_rowwise_data` + `_rowwise_scale_inv`) and columnwise (`_columnwise_data` + `_columnwise_scale_inv`) planes. Previously, MFSDPv2 gathered both planes on every unshard even though `Phase.FORWARD` (`Y = X * W^T`) only consumes rowwise planes and non-recompute `Phase.BACKWARD` (`dX = dY * W`) only consumes columnwise planes.
   - In TransformerEngine, Triton `_row_id_map_pass_2_kernel` lacked a capacity clamp when `total_routed > num_out_tokens` under `drop_and_pad`, writing out of bounds past `num_out_tokens`.
2. **Cross-Rack Expert All-Gather in TorchTitan 816B-A42B GB300 RL & Flux Unpacked Schedule Shift**:
   - Running the Muse Spark 816B-A42B RL trainer on `256` GB300 GPUs (`FSDP=64 x TP=4 x EP=4`) forced expert FSDP all-gathers across `64` ranks spanning multiple racks; meanwhile, Flux `denoise` called `get_schedule` before `pack_latents`, passing a `4x` inflated unpacked sequence length (`latent_height * latent_width` instead of packed `latents.shape[1]`).
3. **Out-of-Order Lookup Races & Tail Lock Leaks in LMCache**:
   - Polling-driven `QUERY_PREFETCH_STATUS` in `LMCacheAsyncLookupClient` added a 1-scheduler-step delay, could arrive at a storage server before `LOOKUP` completed (causing false `0`-hit misses and leaked locks), and failed to release locks for chunks in `[min_hits, server_hits)` when storage servers returned unequal hit counts.
4. **Per-WR `clock_gettime` Syscall Overhead Under Lock in Mooncake RDMA Post Chains**:
   - `RdmaEndPoint::submitPostSend` called `getCurrentTimeInNano()` (`~19 ns` per WR) for every slice while holding `lock_`, dominating chain construction time (`6.3 us` for 256 WRs) and delaying `ibv_post_send` doorbells.
5. **False Sharing in NIXL `sharedRingBuffer` & Per-Token Shared Atomic Contention in Dynamo**:
   - NIXL placed `write_pos` and `read_pos` on adjacent words and loaded the remote atomic on every `push`/`pop`; Dynamo `RequestMetrics` executed a shared atomic `fetch_add(1)` on every generated token across all concurrent requests.

---

## 2. Upstream Implementation & Mechanism Analysis

### 2.1 Megatron-LM MFSDPv2 Phase-Specific MXFP8 Planes (`#7907`) & TE Fixes (`#3347`, `#3701`)

- **Phase-Aware MXFP8 Unsharding (`18bc49d2f6bc`)**:
  - Tracks per-phase unshard events (`_unshard_events: dict[Phase, Event]`) and gathers only rowwise planes in `Phase.FORWARD` and columnwise planes in non-recompute `Phase.BACKWARD`, cutting per-phase MXFP8 all-gather payload and unsharded plane buffer memory by **50.0%**.
  - `update_mxfp8_tensor` pads scales only for newly gathered non-empty planes so it never overwrites the other phase's already-padded scales.
- **TE Triton Capacity Drop & `GroupedTensor` View/Stream Tracking (`c9c92255465b`, `2199b7166417`)**:
  - `_row_id_map_pass_2_kernel` applies `row_id = tl.where(row_id >= num_out_tokens, -1, row_id)` to drop over-capacity tokens safely.
  - `GroupedTensor` preserves autograd `_base` linkage via `_make_wrapper_subclass` + `view_as` and forwards `record_stream` across all underlying data, scale, and amax tensors.

### 2.2 TorchTitan: GB300 128-GPU RL Topology (`#5581`) & Flux Packed-Latent Schedule (`#5580`)

- **Muse Spark 816B-A42B GB300 RL Trainer on 128 GPUs (`43d1bb98014e`)**:
  - Transitions from `256` GPUs (`FSDP=64 x TP=4 x EP=4`) to **`128` GPUs (`FSDP=8 x TP=4 x CP=4 x EP=16`)**, keeping each `EP=16` group within a single 4-host GB300 NVLink domain, shrinking expert FSDP all-gather groups by **`8.0x`** (`64 -> 8`), and enabling `UlyssesVarlenInnerAttention` on both global and sliding-window layers (`scratch_capacity_factor=16.0`), raising `131,072`-context MFU from `15%` to **`28%`** (`186 GiB` peak VRAM).
- **Flux `denoise` Packed Sequence Length Fix (`d681f48b43ad`)**:
  - Moves `get_schedule` after `pack_latents` and passes `latents.shape[1] = (H/16)*(W/16)` (`256` at 256px instead of `1024`).

### 2.3 LMCache (`#5960`, `#5922`), Mooncake (`#4793`), NIXL (`#2018`), & Dynamo (`#17067`, `#16106`)

1. **LMCache Callback-Driven `LookupState` & Excess Tail Lock Release (`ff9185d419e1`, `a81b4e5aea0c`)**:
   - Dispatches `QUERY_PREFETCH_STATUS` immediately inside each server's `LOOKUP` completion callback, caps re-queries to at most 1 per scheduler check (`scheduler_checked`), and calls `free_lookup_locks` on `[min_hits, server_hits)` for servers reporting more hits than `min(hits)`. `PrefetchTaskSpec` adds separate `allowed_l1_memory_tags` and `allowed_l2_memory_tags`.
2. **Mooncake Single Clock Read Per RDMA Post Chain (`dd1217b6c461`)**:
   - Reads `post_ts = getCurrentTimeInNano()` once per QP chain outside the WR loop under `lock_`, cutting 256-WR (2×128) chain build latency from `6.3 us` to **`1.9 us`** (`25 ns -> 7.5 ns` per WR, **`-69.84%`**).
3. **NIXL 256B-Aligned RingBuffer (`a9f1f7a76f46`) & Dynamo 64-Token Counter Batching (`906e89d3a304`)**:
   - NIXL (`TELEMETRY_VERSION = 6`) aligns `write_pos` (offset `0`) and `read_pos` (offset `256`) to `DESTRUCTIVE_INTERFERENCE_SIZE = 256` bytes (`sizeof(bufferHeader) == 512`), caches `cachedReadPos_` / `cachedWritePos_` locally so `push`/`pop` only read the remote atomic when full/empty, and uses `flock(LOCK_SH)` for consumer liveness.
   - Dynamo `RequestMetrics` accumulates per-token increments in request-local `pending_output_tokens` and flushes to the shared model counter every `LOCAL_FLUSH_TOKENS = 64` tokens (and on `finish`/`Drop`), reducing shared atomic contention by **`64.0x`**, alongside `--freeze-gc-after-init`.

---

## 3. Deterministic Verification Summary (`own-offline`)

Verified by `python experiments/megatron-mfsdp-mxfp8-planes-lmcache-lookup-nixl-ring/run.py --check`:

| Metric / Invariant | Baseline / Before | Optimized / After | Deterministic Finding |
|---|---|---|---|
| MFSDPv2 MXFP8 per-phase all-gather (`4096x4096`) | `34,603,008 B` (dual plane) | `17,301,504 B` (single plane) | **`-50.0%`** per-phase all-gather & buffer memory |
| TorchTitan 816B GB300 RL trainer | `256` GPUs (`FSDP=64, EP=4`), `15%` MFU | `128` GPUs (`FSDP=8, CP=4, EP=16`), `28%` MFU | **`50%` fewer GPUs**, **`8.0x`** smaller expert FSDP group |
| LMCache multi-server lookup (`hits=[6, 4, 7]`) | Polling lag + leaked `[64..112)` locks | Callback dispatch + frees `srv0[64:96)`, `srv2[64:112)` | Matches `64` tokens with zero tail lock leaks |
| Mooncake 256-WR (`2x128`) chain build under lock | `256` clock calls / `6.3 us` | `2` clock calls / `1.9 us` | **`-69.84%`** lock hold & doorbell delay |
| NIXL RingHeader & Dynamo 500-token shared atomics | False sharing / `500` shared atomic adds | `512 B` header (`256 B` aligned) / `8` flushes | `0` remote loads on 7 non-full pushes; **`64.0x`** fewer shared atomics |
