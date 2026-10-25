# FlashAttention-4 SM90/SM100 Ragged TMA Load Bounding & `M=64` 2-CTA Backward, verl SGLang MXFP8/MXFP4/NVFP4 Refit, Megatron EP A2A Segment Recompute, Mooncake HCA Affinity & Lock-Free Eviction, NIXL Telemetry, Dynamo GC Freeze & TE Compiled FP8 Caching

Deterministic Python 3.12 standard-library model auditing ten first-hand upstream commits across FlashAttention-4 (`#3012`, `#3013`, `#2965`), verl (`#8306`), Megatron-LM (`#6429`), Mooncake (`#4631`, `#4854`), NIXL (`#2478`), Dynamo (`#16106`), and TransformerEngine (`#3505`):

1. **FlashAttention-4 SM90/SM100 Ragged K/V TMA Load Bounding & `M=64` 2-CTA TMEM Layout (`#3012`, `#3013`, `#2965`)**:
   - Models load-safe wraparound TMA descriptors bounded by `seqlen_k` (`cu_seqlens_k` / `seqused_k`), eliminating `0 * NaN = NaN` `dQ` poisoning when uninitialized K/V tail rows `[seqlen_k, tile_n)` contain `NaN`.
   - Models SM100 2-CTA `tile_n = 64` (`M=64` TMEM layout where each TMEM lane holds half a row and each accumulator uses `N / 2` columns, staging `P^T` and `dS^T` in SMEM), fitting `dK + dV + S + dP + dQ = 128 + 128 + 32 + 32 + 128 = 448 <= 512` TMEM columns at `hdim = 256` without spilling.
2. **verl SGLang Online Weight Refit for MXFP8, MXFP4 & NVFP4 Rollouts (`#8306`)**:
   - Models bucket-aware canonical staging + post-final-bucket in-place kernel layout reconstruction (preserving CUDA-graph-captured pointers) on Qwen3.5-35B-A3B (Megatron EP=8 -> SGLang TP=2 on Blackwell), reducing trainer-rollout `k3_kl` divergence from `0.00571 -> 0.00200` (`2.86x` reduction) for MXFP4 MoE and `0.00946 -> 0.00653` (`1.45x` reduction) for NVFP4 MoE with matching QAT.
3. **Megatron-Core Per-Layer-Segment EP A2A Overlap Activation Recompute (`#6429`)**:
   - Models `ModelChunkState` pre-segment snapshot (`input_ids`, `position_ids`, `mtp_input_mask`) + `reset_for_recompute()` reference release during 1F1B EP A2A overlap, cutting peak chunk activation memory across `8` MoE layers per PP stage from `8.0x` to `1.88x` single-layer footprint (`4.27x` reduction) while preventing double-rolled `mtp_input_mask` corruption.
4. **Mooncake HCA/NUMA Affinity (`#4631`) & Shard-Decoupled `CursorSlice` Eviction (`#4854`)**:
   - Models `MC_STORE_HCA_AFFINITY=1` receiver-local HCA/NUMA segment selection and `CursorSlice` disjoint read-cursor iteration + `DeferDeletion` outside `kNumShards = 1024` metadata write locks (`12.8x` reduction in shard write-lock hold time during DFS bucket eviction).
5. **NIXL Repost/Release Telemetry (`#2478`), Dynamo `--freeze-gc-after-init` (`#16106`), and TE `torch.compile` FP8 Weight Caching (`#3505`)**:
   - Quantifies NIXL exact-once completion accounting (`0%` dropped repost/release completions), Dynamo `gc.freeze()` eliminating `~1,000,000` static heap object walks per Gen-2 GC cycle, and TE compiled `is_first_microbatch` workspace reuse (`539 MiB` stable peak across 4x4096 layers).

## Usage

```bash
python experiments/fa4_sm90_sm100_ragged_tma_m64_bwd_verl_sglang_refit_megatron_ep_recompute_mooncake_nixl_dynamo_te/run.py --check
```
