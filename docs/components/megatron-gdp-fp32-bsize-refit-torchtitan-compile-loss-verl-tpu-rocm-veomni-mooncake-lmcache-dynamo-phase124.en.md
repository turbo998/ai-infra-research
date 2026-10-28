# Megatron-LM GDP FP32 Sigmoid & Batch-Size Ramp / Refit Balance, TorchTitan LocalCompileConfig & Generic Loss, veRL TPU v7x / ROCm DSV4 / VeOmni FSDP2 Offload, Mooncake EFA Stall Floor / WeakPtr Handle / NUMA Cpuset / HCCS 2MiB Alignment, LMCache & Dynamo Deep Dive (Phase 124)

> **中文版**: [megatron-gdp-fp32-bsize-refit-torchtitan-compile-loss-verl-tpu-rocm-veomni-mooncake-lmcache-dynamo-phase124.md](megatron-gdp-fp32-bsize-refit-torchtitan-compile-loss-verl-tpu-rocm-veomni-mooncake-lmcache-dynamo-phase124.md)
> **Evidence Tier**: `implementation` + `author-benchmark` + `own-offline` (Zero GPU hardware re-measurement claims)

---

## 1. Architectural Context & Problem Statement

Across hybrid linear-attention pretraining (GatedDeltaProduct), post-training RL (TPU v7x, AMD ROCm MI300X/MI350X, VeOmni FSDP2), and RDMA/EFA/HCCS KV transport engines, Phase 124 Track C resolves nine foundational engineering bottlenecks:

1. **Zeroed Saturation Gradients in Megatron-LM `GatedDeltaProductMixer` (`a296523510`)**: Computing `b.sigmoid()` directly in BF16 rounds $\sigma(b)$ to `1.0` for $b \in [6.25, 9.0]$ ($\sigma(6.25) \approx 0.99807328 \to 1.0_{\text{BF16}}$), causing backward $\beta(1-\beta)$ to vanish to `0.0` instead of preserving the non-zero FP32 gradient ($2\sigma(1-\sigma) = 0.00384602$).
2. **Missed Batch-Size Ramp Transitions at Large DP & Single-DP-Group Refit Bottleneck (`d1eec51b85`, `4a55e6fee6`)**: At `DP = 8192`, ramping `global_batch_size` from `8192` to `12288` kept integer-floor `num_microbatches == 1`, bypassing `is_batch_size_transition`; `_round_robin_dp` returned `src_meta_list[0]` whenever `len(grouped_by_dp) == 1`, concentrating all tensor transfers onto replica 0.
3. **Out-of-Tree Compile Hooks & Coupled Loss Types in TorchTitan (`0f9fd2bd64`, `740d5f755a`)**: Models needed an extensible `LocalCompileConfig` and a contravariant `Loss[LossInputT]` protocol decoupling `MTPLoss` and `ChunkedLossWrapper` from `CrossEntropyLoss`.
4. **veRL TPU v7x Qwen3.5 GRPO Precision, ROCm DeepSeek-V4 B-Preshuffle & VeOmni FSDP2 Host Peak Memory (`2ab3529203`, `6d04f398f1`, `b65da71c17`, `6f3f449e11`)**: Under FSDP2 CPU offload, dropping `override_optimizer_config` prevented fused AdamW from running, allocating a full parameter-sized FP32 temporary buffer on host RAM (**`130.39 GiB`** for a 35B model).
5. **Mooncake EFA Premature QP Resets, `OffsetAllocationHandle` Atomic Contention, Container `cpuset.mems` `mbind` Failures & Ascend HCCS Alignment (`3a05eec5ef`..`5ef59b0c6e`)**: `kMaxStalledWaves = 1024` triggered QP resets during brief microbursts without a wall-clock floor; copying `shared_ptr` per allocation incurred 2 atomic refcount operations; `mbind(MPOL_BIND)` failed with `EINVAL` when NIC NUMA nodes lay outside container `cpuset.mems`; Ascend HCCS required `2 MiB` device memory alignment.
6. **LMCache All-Zero False Positives & Dynamo Unbounded DP Queue Bloat (`58c2ace2c2`, `0108f5b634`, `cd3b977edb`, `08907b7ad7`)**: Zero-filled KV buffers could not distinguish real cache hits from uninitialized memory, and Dynamo allowed `20,000` queued requests per DP rank (`160,000` across `DP=8`).

---

## 2. Implementation Mechanics & Quantitative Evidence

### 2.1 Megatron-LM FP32 Sigmoid Gating, Batch Ramp Detection & Refit Load Balance
- **FP32 Sigmoid Gradient Preservation (`a296523510`)**: `beta = b.float().sigmoid().to(b.dtype)` forces PyTorch Autograd to save the FP32 sigmoid output (`0.99807328` at $b=6.25$), preserving non-zero backward gradients ($0.00384602$ at $b=6.25$ and $0.00024679$ at $b=9.0$ vs `0.0` in BF16).
- **Batch-Size Transition & Refit Round-Robin (`d1eec51b85`, `4a55e6fee6`)**: Checks `get_current_global_batch_size() != global_batch_size` under `has_step_batch_size_schedule` and removes the `len(grouped_by_dp) == 1` fast path in `_round_robin_dp`, cutting peak per-replica tensor load by **`8.0x`** across 8 replicas.

### 2.2 TorchTitan Compile/Loss Contracts & veRL TPU v7x / ROCm / VeOmni FSDP2
- **TorchTitan (`0f9fd2bd64`, `740d5f755a`)**: Adds `LocalCompileConfig` (`model.local_compile.apply()`) and generic contravariant `Loss[LossInputT]`.
- **veRL (`2ab3529203`, `6d04f398f1`, `b65da71c17`, `6f3f449e11`)**: Adds TPU v7x Qwen3.5-35B-A3B GRPO with `--xla_tpu_bf16_emission_mode=PROMOTE_F32`, ROCm (`gfx942`/`gfx950`) DeepSeek-V4 rollout (`VLLM_ROCM_DSV4_BPRESHUFFLE=0`), and VeOmni FSDP2 CPU offload honoring `override_optimizer_config` (eliminating **`130.39 GiB`** of host FP32 temporary allocations on 35B models) plus `ep_sharded_stream_load` and trainer-side `Fp8ConfigWeightQuantizer`.

### 2.3 Mooncake Transport Hardening, LMCache Deterministic KV & Dynamo DP Queue Cap
- **Mooncake (`3a05eec5ef`..`5ef59b0c6e`)**:
  - Requires both `1,024` stalled waves and `efa_stall_min_ms = 1000 ms` (`50 ms` floor, `MC_EFA_STALL_MIN_MS`) before EFA QP reset.
  - Replaces `shared_ptr` with `weak_ptr` in `OffsetAllocationHandle` (`2 -> 0` atomic refcount ops/alloc).
  - Intersects NIC NUMA nodes with `cpuset.mems` and task `MPOL_BIND` before `mbind`, enforces `2 MiB` (`2,097,152 B`) Ascend HCCS alignment, exposes `max_blocks_per_context`, and prevents torn `activeRanks` reads in `applyGroupState`.
- **LMCache & Dynamo (`58c2ace2c2`, `0108f5b634`, `cd3b977edb`, `08907b7ad7`)**: Seeds cold/warm KV verification buffers in `[1, 127]` via `SHA-256(req_id:layer_id)`, introduces immutable `ObjectKeyIdentity`, and divides Dynamo's `40,000` admission cap across DP ranks (`5,000`/rank at `DP=8`, **`4.0x`** reduction vs `20,000`/rank).

---

## 3. Deterministic Verification

- **Verification Script**: `experiments/megatron_gdp_fp32_bsize_refit_torchtitan_compile_loss_verl_tpu_rocm_veomni_mooncake_lmcache_dynamo_phase124/run.py`
