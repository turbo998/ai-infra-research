# Deep Dive: Megatron-LM Step-Wise LR Multipliers / Batch-Invariant Vocab-Parallel RL Logprobs + PyTorch ROCm Mixed-Dtype Vectorized Kernel Precision / Pickler `pushGlobal` Memoization + Mooncake Topology-Local EFA NIC Registration / Atomic-Free HCA QP Setup + LMCache L2 `extra_attributes` + Dynamo Opt-In `x-bypass-remote-prefill` + NIXL Mooncake Batch Sizing & GPUNetIO 6-Bug Fix (Phase 134 · Priority 143)

## 1. Problem Scope & Distributed Infrastructure Overview

This report addresses `q-training-megatron-steplr-rl-logprobs-pytorch-rocm-pickler-mooncake-efa-lmcache-dynamo-nixl-phase134` (Priority 143) using 10 zero-overlap upstream commits merged between 2026-10-29 and 2026-10-31 across `NVIDIA/Megatron-LM`, `pytorch/pytorch`, `kvcache-ai/Mooncake`, `LMCache/LMCache`, `ai-dynamo/dynamo`, and `ai-dynamo/nixl`.

---

## 2. Primary Source Implementation Analysis

1. **Megatron-LM Step-Wise LR Multipliers with Linear Ramps (`NVIDIA/Megatron-LM@a2946a0`)**: Adds `step_lr_rescale_schedule` (`SAMPLES:MULTIPLIER`) and `step_lr_rescale_warmup_samples` to smoothly scale LR across dynamic batch-size transitions.
2. **Megatron-LM Batch-Invariant Vocab-Parallel RL Logprobs (`NVIDIA/Megatron-LM@1ac85e9`)**: Makes `_VocabParallelSelectiveLogProbs` bitwise batch-invariant (`0.0` drift across microbatch splits) while retaining TP vocab sharding (`75%` logit memory saved at TP=4).
3. **PyTorch ROCm Mixed-Dtype Vectorized Elementwise Kernel Precision (`pytorch/pytorch@81e2049`)**: Loads vectorized inputs into the functor's `arg_t` (`fp32`) instead of rounding early to the output dtype (`bf16`/`fp16`), reducing `torch.add(x_bf16, y_f32, out=out_bf16)` mismatch at `n=65536` from `29.0% -> 0.0%`.
4. **PyTorch Pickler `pushGlobal` Memoization (`pytorch/pytorch@4ec7a29`)**: Caches `<Dtype>Storage` global memo IDs in `storage_type_memo_ids_`, emitting `pushBinGet` on subsequent tensors (`47.0 -> 20.0 ms`, `2.35x` faster serialization).
5. **Mooncake Topology-Local EFA NIC Default Registration (`kvcache-ai/Mooncake@6c83ea2`)**: Sets `MC_EFA_NIC_SELECTION=local` by default on `p6-b300.48xlarge` (16 EFA NICs, 2 per GPU), cutting vLLM TP8 KV-cache registration on the slowest rank from **`274.0 -> 9.0 s` (`30.44x`)** and 1P1D readiness from **`822.0 -> 308.0 s` (`2.6688x`)** while preserving `761.8 GB/s` GPU-to-GPU bandwidth.
6. **Mooncake TE/TENT Atomic-Free HCA QP Setup (`kvcache-ai/Mooncake@956af65`)**: Omits `IBV_ACCESS_REMOTE_ATOMIC` when `device_attr.atomic_cap == IBV_ATOMIC_NONE` and honors explicit TENT read depth up to hardware limits.
7. **LMCache Request-Scoped `ObjectKey.extra_attributes` for L2 Adapters (`LMCache/LMCache@71a826f`)**: Passes non-identity metadata (such as NUMA placement) to L2 adapters without polluting cache key hashes (`1.45x` local NUMA load throughput).
8. **Dynamo Opt-In `x-bypass-remote-prefill` SSE Marker (`ai-dynamo/dynamo@fc78d7c`)**: Emits the annotation-only SSE frame only when requested in `nvext.annotations`, eliminating `100% -> 0%` stream aborts in OpenAI Python/Node SDKs.
9. **NIXL `MOONCAKE` Plugin Naming & Descriptor-Count Batch Sizing (`ai-dynamo/nixl@2170648`)**: Sizes Mooncake batches to exact descriptor counts and standardizes the `MOONCAKE` backend identifier.
10. **NIXL DOCA GPUNetIO 6-Bug Fix & Ring Readback Removal (`ai-dynamo/nixl@a4565f5`)**: Tracks per-peer notification CQ indices, bounds `cuda_streams` to `[1, 4]` to prevent memory corruption of `wait_stream`/`postLock_`, and removes GPU request-ring readback in `postXfer()` (`6.8 -> 2.1 us`, `3.2381x`).

---

## 3. Reproducible Offline Verification

- **Command**: `python experiments\megatron_steplr_rl_logprobs_pytorch_rocm_pickler_mooncake_efa_lmcache_dynamo_nixl_phase134\run.py --check`
- **Artifact**: `experiments/megatron_steplr_rl_logprobs_pytorch_rocm_pickler_mooncake_efa_lmcache_dynamo_nixl_phase134/results/summary.json`
