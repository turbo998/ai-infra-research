# Architectural Audit: FlashAttention-4 SM100 Backward Pack-GQA & Ragged TMA Descriptor Bounds, veRL Single-Block Host LoRA Staging & Mooncake EFA Stale Peer Eviction

> **中文版**: [fa4-sm100-bwd-pack-gqa-ragged-tma-desc-verl-lora-host-stage-mooncake-efa.md](fa4-sm100-bwd-pack-gqa-ragged-tma-desc-verl-lora-host-stage-mooncake-efa.md)

## 1. Context & Architectural Challenges

Across Blackwell SM100 attention backward passes, RLHF/GRPO MoE LoRA weight synchronization, and 240-node disaggregated serving rolling upgrades, four systems bottlenecks govern training throughput, memory safety, and transport liveness:
1. **FA4 SM100 Backward GQA Atomic Reduction Overhead vs. Ragged Serialization**: Running one backward tile per query head requires reducing `dK / dV` across the GQA group in FP32 (`cp.reduce.async.bulk` + postprocess). Folding `qhead_per_kvhead` query heads into the rows of the Q tile (Pack-GQA) keeps `dK / dV` in TMEM and takes the direct store path, but multiplies the longest CTA's loop count by `qh`—causing a **1.72x regression** (`3946 µs` packed vs. `2297 µs` unpacked on B200) on ragged varlen packs with one 8k sequence and 16 short sequences (`MQA 16/1 hd256`).
2. **FA4 Ragged K/V TMA ~128 KB Over-Prefetch & Padded-Grid Workspace Overflow**: The 5-D wrap-around TMA descriptor used to bound `seqlen_k` had an unbounded footprint, allowing the hardware TMA unit to prefetch up to **~128 KB** past a partially out-of-bounds box and fault (`cudaErrorIllegalAddress`) when unmapped memory followed V or K. Furthermore, sizing the tensormap workspace without the varlen scheduler's per-batch padding allocated only ~300–302 slots for `B=49, seqlen_q=128, H=6` on H100 while the scheduler launched `582` CTAs.
3. **veRL MoE LoRA IPC Sync Allocation Storm**: Syncing per-expert MoE LoRA adapters (`16,000` tensors of `(16, 1024)`, 500 MiB) via `update_weights_from_ipc` previously cloned every tensor twice on GPU and allocated/pinned each tensor individually on host in `LoRAModel.from_lora_tensors`, taking `705 ms` per bucket on GB300.
4. **Mooncake EFA Address-Handle Exhaustion on Rolling Upgrades & NIXL SPSC False Sharing**: During rolling upgrades of `120 prefill + 120 decode` nodes, volatile port/timestamp fields in `peer_nic_path` leaked stale entries in `peer_map_`, exhausting libfabric's internal address-handle (AH) cache (`fi_av_insert` returning 0). Meanwhile, NIXL's `sharedRingBuffer` placed producer and consumer atomics on the same cache line and re-read remote shared-memory positions on every push/pop.

---

## 2. First-Hand Source Code Audit

### 2.1 FA4 SM100 Backward Pack-GQA & Critical-Path Heuristic (`Dao-AILab/flash-attention` @ `bc52a40aef`)
Auditing `flash_attn/cute/flash_bwd_sm100.py`, `flash_bwd_preprocess.py`, and `flash_bwd_postprocess.py`:
- **Direct TMEM `dK / dV` Store**: Row `r` of the packed sequence maps to head `r % qh` at token position `r // qh`. One tile per KV head loads K/V once, accumulates `dK / dV` in TMEM across the GQA group, and writes them directly without FP32 atomics, postprocess kernels, or `dK/dV` semaphores.
- **Two-Stage `_bwd_pack_gqa_heuristic`**:
  1. **Critical-Path Guard**: Rejects packing when the longest packed CTA's iterations `qh * ceil(seqlen_q / tile_m)` exceed the average work per cluster slot (computed over mean sequence lengths for varlen), preventing the `3946 µs` vs. `2297 µs` ragged MQA serialization on B200;
  2. **Wave-Quantization Guard**: Compares `gain ~ 0.25 * 2048 / seqlen_q` against `(ceil(w) - w) / ceil(w)`. On B200 (1400 MHz), `hd128` GQA 2k improves by **+13%..+26%** and `hd256` 2k/8k improves by **+4%..+11%**.

### 2.2 Per-CTA TMA Descriptor Updates & Padded Grid Sizing (`10b74d98fb`)
Auditing `flash_attn/cute/flash_bwd_sm100.py` and `flash_fwd_sm100.py`:
- **Bounded Per-CTA TensorMap Updates**: Replaces `create_ragged_tensor_for_tma_load` (5-D wrap-around) with `cutlass.utils.TensorMapManager` (`GMEM` mode), allocating one 128-byte slot per descriptor per CTA (`3 × 128 = 384 B/CTA` in backward for K, V, Kt) and fencing/prefetching before both dense and block-sparse producer loads.
- **Padded Grid Bound**: Sizes the descriptor workspace to `(rows // tile + 2 * batch + 1) * heads` (`888` slots for `B=49, seqlen_q=128, H=6, tile=128`, safely covering all `582` launched CTAs on H100) with an in-kernel bounds trap.

### 2.3 veRL Single-Host-Block LoRA Staging (`3c6d9b43b6`), Mooncake EFA (`487da6af82`) & NIXL SPSC (`a9f1f7a76f`)
- **veRL Single-Block Pinned Staging (`3c6d9b43b6`)**: `verl/workers/rollout/vllm_rollout/utils.py` allocates one pinned CPU block in `lora_dtype` per bucket, copies/casts each IPC tensor directly into its slice, and hands out views that pass through vLLM's `.to("cpu", dtype).pin_memory()` as no-ops—cutting 250 MiB (`4,000` tensors) from `165 ms` to **`35 ms` (4.71x)** and 500 MiB (`16,000` tensors) from `705 ms` to **`131 ms` (5.38x)** on GB300.
- **Mooncake EFA (`487da6af82`) & NIXL SPSC (`a9f1f7a76f`)**: Evicts stale peer endpoints on EFA reconnect to keep active libfabric AH entries bounded at `240` across `120P + 120D` rolling upgrades; pads NIXL SPSC ring-buffer atomics to separate cache lines with cached remote indices.

---

## 3. Deterministic Verification (`fa4_sm100_bwd_pack_gqa_ragged_tma_desc_verl_lora_host_stage_mooncake_efa`)

| Subsystem / Metric | Baseline Behavior | Phase 81 Audited Path | Verified Result | Source ID |
|---|---|---|---|---|
| FA4 SM100 Bwd Pack-GQA (`hd128` 2k / Ragged MQA) | Unpacked (`410 µs`) / Naive Pack (`3946 µs`) | Critical-Path Guard + TMEM `dK/dV` (`325.4 / 2297 µs`) | **+26% Speedup & Avoids 1.72x Regression** | `src-fa4-sm100-bwd-pack-gqa-critical-path-2026` |
| FA4 Ragged TMA Workspace (`B=49, S=128, H=6`) | 5-D Wrap (~128KB over-read) + `300` slots (< `582` CTAs) | Per-CTA GMEM TensorMap + `888` bounded slots | **0 Faults / 100% Covers 582 CTAs** | `src-fa4-ragged-kv-tma-per-cta-descriptor-2026` |
| veRL MoE LoRA Sync (`16,000` tensors, 500 MiB, GB300) | 2× GPU clones + 16k pinned allocs (`705 ms`) | Single Pinned Host Block + Zero-Copy Views (`131 ms`) | **5.38x Speedup (16,000 -> 1 alloc)** | `src-verl-vllm-lora-bucket-single-host-block-2026` |
| Mooncake EFA Rolling Upgrade (`120P + 120D` nodes) | Stale peer leak (AH exhausted after 17 waves) | Evict stale peer on EFA reconnect | **Bounded at 240 active AHs** | `src-mooncake-efa-evict-stale-peer-reconnect-2026` |
