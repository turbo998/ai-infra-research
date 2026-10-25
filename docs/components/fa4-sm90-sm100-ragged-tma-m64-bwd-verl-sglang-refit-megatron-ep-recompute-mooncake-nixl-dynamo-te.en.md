# FlashAttention-4 SM90/SM100 Ragged TMA Load Bounding & `M=64` 2-CTA Backward, verl SGLang MXFP8/MXFP4/NVFP4 Weight Refit, Megatron EP A2A Segment Recompute, and Mooncake / NIXL / Dynamo / TE Co-Design

> **中文版本**: [fa4-sm90-sm100-ragged-tma-m64-bwd-verl-sglang-refit-megatron-ep-recompute-mooncake-nixl-dynamo-te.md](fa4-sm90-sm100-ragged-tma-m64-bwd-verl-sglang-refit-megatron-ep-recompute-mooncake-nixl-dynamo-te.md)  
> **Evidence ID**: `ev-fa4-sm90-sm100-ragged-tma-m64-bwd-verl-sglang-refit-megatron-ep-recompute-mooncake-nixl-dynamo-te-2026`  
> **Experiment**: [`experiments/fa4_sm90_sm100_ragged_tma_m64_bwd_verl_sglang_refit_megatron_ep_recompute_mooncake_nixl_dynamo_te/run.py`](../../experiments/fa4_sm90_sm100_ragged_tma_m64_bwd_verl_sglang_refit_megatron_ep_recompute_mooncake_nixl_dynamo_te/run.py)

---

## 1. First-Hand Source Audit Scope

This document audits ten first-hand upstream commits across `Dao-AILab/flash-attention`, `volcengine/verl`, `NVIDIA/Megatron-LM`, `kvcache-ai/Mooncake`, `ai-dynamo/nixl`, `ai-dynamo/dynamo`, and `NVIDIA/TransformerEngine` (`#3012`, `#3013`, `#2965`, `#8306`, `#6429`, `#4631`, `#4854`, `#2478`, `#16106`, `#3505`):

1. **FlashAttention-4 SM90/SM100 Ragged K/V TMA Load Bounding & `M=64` 2-CTA TMEM Layout (`#3012` `2bdeddd15a`, `#3013` `767386d814`, `#2965` `2b00d76f22`)**: Bounds K/V TMA descriptors by logical `seqlen_k` so uninitialized `NaN` tail rows cannot poison `dQ` via `0 * NaN = NaN`, and introduces the `M=64` 2-CTA TMEM layout fitting `hdim=256` backward into `448 <= 512` TMEM columns.
2. **verl SGLang Online Weight Refit for MXFP8, MXFP4 & NVFP4 Rollouts (`#8306`, `5265d561a3`)**: Stages canonical checkpoint tensors across sync buckets and reconstructs backend-specific kernel layouts in-place after the final bucket without invalidating CUDA-graph-captured pointers.
3. **Megatron-Core Per-Layer-Segment EP A2A Overlap Activation Recompute (`#6429`, `da2c0ec614`)**: Releases per-segment forward activation references immediately via `reset_for_recompute()` while snapshotting `ModelChunkState` (including `mtp_input_mask`).
4. **Mooncake HCA/NUMA Affinity (`#4631`, `868a6d05ab`) & Shard-Decoupled `CursorSlice` Eviction (`#4854`, `c81f4ad486`)**: Moves DFS bucket file deletion and soft-pin expiry off the `kNumShards = 1024` metadata write locks.
5. **NIXL Repost/Release Telemetry (`#2478`, `70badd6aad`), Dynamo `--freeze-gc-after-init` (`#16106`, `45cc9e8a1a`), and TE `torch.compile` FP8 Weight Caching (`#3505`, `9ea05cf2e8`)**.

---

## 2. Implementation Paths & Architectural Derivations

### 2.1 FlashAttention-4 Ragged K/V TMA Bounding & `M=64` 2-CTA TMEM Layout (`#3012`, `#3013`, `#2965`)

- **Eliminating `0 * NaN = NaN` `dQ` Poisoning (`#3012`, `#3013`)**: Because FA4 backward is KV-stationary and loads whole `tile_n` K/V tiles via TMA, unused rows in `[seqlen_k, tile_n)` (under `cu_seqlens_k` or `seqused_k`) arrive with whatever uninitialized values reside in the buffer. Even though `dS` is masked to `0.0`, `dP = dO @ V^T` and `dQ = dS @ K` multiply `0.0 * NaN = NaN`, poisoning `dQ` for any sequence whose last K tile straddles the tail. Building load-safe ragged TMA descriptors bounded by `(offset_k, seqlen_k)` leverages hardware TMA out-of-bounds zero-fill with zero pipeline or barrier overhead.
- **SM100 `M=64` 2-CTA TMEM Layout (`#2965`)**: At `hdim = 256`, a standard `tile_n = 128` layout requires $256 + 256 + 128 + 128 + 256 = 1024$ TMEM columns, exceeding the `512`-column SM100 hardware limit. With `tile_n = 64` in 2-CTA mode (`M = 128` pair MMA), each TMEM lane holds half a row so every accumulator occupies $N / 2$ columns while `P^T` and `dS^T` are staged in SMEM:
  $$\text{Cols}_{\text{TMEM}} = 128 (dK) + 128 (dV) + 32 (S) + 32 (dP) + 128 (dQ) = 448 \le 512$$

### 2.2 verl SGLang MXFP8 / MXFP4 / NVFP4 Online Weight Refit (`#8306`)

In `verl/utils/sglang/`, a bucket-aware custom loader stages MXFP8 scales and MXFP4/NVFP4 routed-expert tensors across all sync buckets and rebuilds DeepGEMM / FlashInfer CUTLASS layouts in-place after the final bucket, preserving CUDA Graph storage. On Blackwell with Qwen3.5-35B-A3B (Megatron EP=8 $\rightarrow$ SGLang TP=2), matching QAT reduces trainer–rollout `k3_kl` from `0.00571` to `0.00200` (**2.86x** lower) for MXFP4 MoE and from `0.00946` to `0.00653` (**1.45x** lower) for NVFP4 MoE.

### 2.3 Megatron-Core Per-Layer-Segment EP A2A Overlap Recompute (`#6429`)

In `model_chunk_schedule_plan.py` and `utils.py`, `reset_for_recompute()` drops `inputs`/`output` references immediately after each recomputed layer segment finishes forwarding (without calling `resize_(0)` on storage so backward replay can rebuild the autograd graph) and snapshots `ModelChunkState` (`input_ids`, `position_ids`, `mtp_input_mask`) so MTP replay does not double-roll `mtp_input_mask`. Across an 8-layer MoE chunk, peak activation memory drops by **4.27x** (`8.0 -> 1.875` layer units).

### 2.4 Mooncake HCA Affinity (`#4631`) & Shard-Decoupled Eviction (`#4854`), NIXL (`#2478`), Dynamo (`#16106`), and TE (`#3505`)

- **Mooncake (`#4631`, `#4854`)**: `MC_STORE_HCA_AFFINITY=1` prioritizes receiver-local HCA/NUMA segments for RL `put`/`get`; `#4854` introduces `CursorSlice` parallel read cursors and moves filesystem `DeleteFile` out of the `kNumShards = 1024` metadata write locks via `CommitEvictionLogical` + `DeferDeletion` (**12.8x** shorter shard write-lock hold time, `640 us -> 50 us`).
- **NIXL (`#2478`), Dynamo (`#16106`), and TE (`#3505`)**: NIXL records completions first observed during `postXferReq` repost checks or `releaseXferReq` without resetting in-flight timers on rejected reposts; Dynamo adds `--freeze-gc-after-init` (`freeze_gc_heap()`) to freeze `~1,000,000` static worker objects before model registration; TE enables `is_first_microbatch` FP8 weight caching under `torch.compile` by allocating a fresh FP8 workspace on the first microbatch and reusing it across remaining microbatches (`539 MiB` peak for 4x4096 layers).

---

## 3. Summary Table

| Framework & Feature | Commit | Parameter / Metric | Baseline | Optimized | Improvement |
|---|---|---|---|---|---|
| FA4 SM90/SM100 Ragged K/V TMA | `#3012` / `#3013` | `seqlen_k` OOB zero-fill | `0 * NaN = NaN` poisons `dQ` | `0.0%` `NaN` `dQ` poison | **100% eliminated** |
| FA4 SM100 `M=64` 2-CTA Backward | `#2965` (`2b00d76f22`) | `tile_n=64`, `hdim=256` TMEM cols | $1024 > 512$ (overflow) | $448 \le 512$ cols | **2.29x col reduction** |
| verl SGLang MXFP4 Refit + QAT | `#8306` (`5265d561a3`) | Qwen3.5-35B-A3B `k3_kl` | $0.00571$ | $0.00200$ | **2.86x lower KL** |
| Megatron EP A2A Segment Recompute | `#6429` (`da2c0ec614`) | 8-Layer MoE chunk activations | $8.0\ \text{units}$ | $1.875\ \text{units}$ | **4.27x lower memory** |
| Mooncake DFS Shard Lock Decouple | `#4854` (`c81f4ad486`) | `kNumShards=1024` write-lock hold | $640.0\ \mu\text{s}$ | $50.0\ \mu\text{s}$ | **12.8x lock reduction** |
