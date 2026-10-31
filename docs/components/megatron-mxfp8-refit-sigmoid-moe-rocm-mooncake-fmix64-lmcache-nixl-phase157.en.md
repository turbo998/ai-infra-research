# Phase 157 Deep Dive: Megatron-LM Single-Copy MXFP8 Expert Refit & Fused Sigmoid Top-K, PyTorch ROCm Static Remainder Kernel, Mooncake fmix64 Lock Shards & Chunk Refcount, LMCache, Dynamo & NIXL Validation

> **Language / 语言**: [English](megatron-mxfp8-refit-sigmoid-moe-rocm-mooncake-fmix64-lmcache-nixl-phase157.en.md) | [中文](megatron-mxfp8-refit-sigmoid-moe-rocm-mooncake-fmix64-lmcache-nixl-phase157.md)

## 1. Overview & Architectural Evolution

This report audits 10 primary commits across `NVIDIA/Megatron-LM`, `pytorch/pytorch`, `kvcache-ai/Mooncake`, `LMCache/LMCache`, `ai-dynamo/dynamo`, and `ai-dynamo/nixl` focusing on single-copy MXFP8 MoE weight refitting, ROCm mixed-dtype elementwise tail specialization, slab-allocated lock-shard hashing in Mooncake TENT, and distributed KV cache metadata safety.

## 2. Primary Source Audit & Quantitative Findings

1. **Megatron-LM Single-Copy FlashInfer MXFP8 Writeback (`dda809c`, `#8560`), Fused Sigmoid Top-K (`faf27c8`, `#8296`) & DDP `math.isclose` (`8b2d9b0`, `#8579`)**: `attach_routed_mxfp8_writeback` writes canonical MXFP8 updates directly into the routed expert stack in place, saving **50.0%** of expert weight HBM (`5.625 GB` vs `11.25 GB` for 32 local experts). `sigmoid_topk.py` collapses 4 compiled router launches into **1** bit-exact Triton launch (**4.0x** launch reduction), while `math.isclose` prevents IEEE-754 rounding assertion failures on non-power-of-two DP sizes.
2. **PyTorch ROCm Statically Typed Remainder (`15ff3c1`, `#202201`) & Pre-Dist `TORCH_TRACE` Rank Naming (`6024c94`, `#202299`)**: Replaces runtime dtype dispatch in `vectorized_templated_elementwise_kernel` remainder blocks with `LoadWithStaticCast` and a 4-element/thread unroll for remainders $\le 2048$, cutting MI300X tail latency from `46.2 us` to `6.1 us` (**7.57x** speedup, **86.8%** reduction) and shrinking `libtorch_hip.so` by `1.60 MB`. Falling back to `RANK` under `TORCHELASTIC_RUN_ID` preserves `rank_<N>_` trace filenames before `init_process_group`.
3. **Mooncake TENT `fmix64` Batch Lock Shard Hashing (`d014478`, `#5095`), Conductor Chunk Refcount (`f5802d4`, `#5112`), LMCache (`#6127`), Dynamo (`#17331`) & NIXL (`#1270`)**: Because $\gcd(\text{sizeof(Batch)}=248, 64) = 8$, raw modulo addressing hit only **8/64 (12.5%)** lock shards; mixing with MurmurHash3 `fmix64` populates **64/64 (100.0%)** shards and boosts 16-thread SHM 4 KiB throughput by **1.88x**. Reference-counting vLLM chunk-mode CPU announcements prevents premature shared-prefix eviction, LMCache splits Ascend KV groups by physical block stride, Dynamo adds the Global Discovery Service for active-active `PoolRelay` streaming, and NIXL validates Libfabric `importStr()` return codes before parsing RDMA descriptors.

## 3. Reproducible Verification

Run `python experiments/megatron_mxfp8_refit_sigmoid_rocm_mooncake_fmix64_nixl_phase157/run.py --check`.
