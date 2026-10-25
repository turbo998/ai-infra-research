# FA4 SM100/SM103 2-CTA dQaccum & hd256 Unification, veRL DeepGEMM QAT Rollout, Megatron Mamba Residual State, Mooncake Soft-Pin & NIXL Trace V2 Deep Dive (Phase 85)

> **Associated Experiment**: `experiments/fa4_2cta_verl_deepgemm_mamba_mooncake_nixl_phase85/`  
> **Associated Question**: `q-fa4-2cta-verl-deepgemm-mamba-mooncake-nixl-phase85` (`priority: 94`)  
> **中文版本**: [`fa4-2cta-verl-deepgemm-mamba-mooncake-nixl-phase85.md`](./fa4-2cta-verl-deepgemm-mamba-mooncake-nixl-phase85.md)

---

## 1. Overview & Cross-Stack Architecture

In Track 3 of Phase 85 (`priority: 94`), we audit 10 first-hand upstream commits across FlashAttention-4, veRL, Megatron-Core, Mooncake Store, NIXL, Dynamo, and TransformerEngine:

1. **FlashAttention-4 SM100/SM103 2-CTA Direct Register `dQaccum`, `hd128` `setmaxnreg.dec` Cap & `hd256` Forward Unification (PR #2960 `47e91f1f`, PR #2950 `037e808b`, PR #2962 `85d6dfe3`)**.
2. **veRL Unified Megatron ModelOpt QAT & DeepGEMM Low-Precision Rollout Policy (PR #8308 `619c397d` & PR #8287 `8f016846`)**.
3. **Megatron-Core `ResidualBranchInput` Mamba Local State & Dynamic `dp_cp` Timeout Propagation (PR #8344 `85bf0a66` & PR #7736 `3630e44a`)**.
4. **Mooncake Store Standalone Batch Soft-Pin HTTP Controls & Per-Key Standby Restore (PR #4858 `e764353f` & PR #4844 `5872ee87`)**.
5. **NIXL Trace Plugin API V2 `TraceContext` Propagation, Dynamo Opt-In `/health` Discovery & TE JAX MoE FSDP Layout Validation (PR #2445 `56c1c2f7`, PR #16454 `a789babb`, PR #3638 `eb3c860a`)**.

---

## 2. Source-Level Mechanics & Quantitative Formulas

### 2.1 FlashAttention-4: 2-CTA `dQaccum` Register Load, `hd128` Crash Fix & `hd256` Unification (`47e91f1f`, `037e808b`, `85d6dfe3`)
- **2-CTA Backward Postprocess Direct Register Load (`47e91f1f`)**: In `flash_bwd_postprocess.py`, the `M=128` 2-SM MMA splits result rows across the CTA pair (`m // 64 = CTA rank`) and places columns `[0, hdim/2)` on lanes `0-63` and `[hdim/2, hdim)` on lanes `64-127`. By defining `_dQaccum_tile_layout_2cta()` over GMEM, `dQaccum` (`128 x 128 x 4 B = 65,536 B` FP32) is loaded straight into registers without staging through SMEM, saving **`65,536 B`** of SMEM (only `32,768 B` BF16 `sdQ` remains in SMEM).
- **SM103 (GB300) `hd128` Decode `setmaxnreg.dec` Trap Fix (`037e808b`)**: At `q_stage = 1`, `num_regs_softmax = 176`, `num_regs_correction = 64`, and `threads_per_cta = 512`, the uncapped formula yielded:
  $$\text{num\_regs\_other}_{\text{uncapped}} = \left\lfloor \frac{512 - 176 \times 1 - 64}{3 - 1} \right\rfloor_{/8 \times 8} = 136 > \frac{65536}{512} = 128$$
  causing `setmaxnreg.dec` to trap with an illegal instruction on GB300. Capping by `65536 // self.threads_per_cta` clamps `num_regs_other` to **`128`**.
- **`hd256` Unification & Unaligned Barrier (`85d6dfe3`)**: Deletes the standalone `2,007`-line `sm100_hd256_2cta_fmha_forward.py`, running `hd256` on the unified `flash_fwd_sm100.py` path with 2-CTA/S ping-pong and `NamedBarrier.arrive_and_wait_unaligned` (`nvvm.barrier_cta_sync(..., aligned=False)`).

### 2.2 veRL: Unified Megatron QAT & DeepGEMM Low-Precision Rollout (`619c397d`, `8f016846`)
- **Trainer-Side Export & DeepGEMM Pinning**: Standardizes `FP8_BLOCK`, `MXFP8`, `MXFP4`, and `NVFP4` around trainer-side `compressed-tensors` export and pins supported vLLM and SGLang Linear/MoE rollout kernels to **`deep_gemm`** (routing `compressed-tensors` MXFP4 experts through DeepGEMM FP4 with dynamic FP8 group-128 activations, while preserving FP32 checkpoint scales on Hopper).
- **Measured Blackwell `k3_kl` Divergence (`Qwen3.5-35B-A3B`, Megatron `EP=8`, vLLM `TP=2`)**:
  - `MXFP8 Linear + MoE`: `0.00387` (no QAT) $\rightarrow$ **`0.00325`** (matching QAT, **`1.191x`** reduction).
  - `BF16 Linear + MXFP4 MoE` (DeepGEMM MoE): `0.00790` (no QAT) $\rightarrow$ **`0.00140`** (matching QAT, **`5.643x`** reduction).
  - `BF16 Linear + NVFP4 MoE` (FlashInfer CUTLASS W4A4): `0.00808` (no QAT) $\rightarrow$ **`0.00722`** (matching QAT, **`1.119x`** reduction).

### 2.3 Megatron-Core: `ResidualBranchInput` & Dynamic `dp_cp` Timeout (`85bf0a66`, `3630e44a`)
- **`ResidualBranchInput[StateT]` (`85bf0a66`)**: Unifies `MambaLayer` and `WideResidualMambaLayer` branch execution by returning `ResidualBranchInput(branch_input, state)` from `_prepare_mixer_state` and passing `state` directly into `_apply_mixer_update`, eliminating instance-attribute side channels.
- **Dynamic `dp_cp` / `expt_dp` Timeout (`3630e44a`)**: Threads `timeout` (from `distributed_timeout_minutes`, e.g., `45 min = 2,700 s`) into `create_dynamic_dp_cp_groups` and `expt_dp` Gloo groups instead of falling back to PyTorch's default `10 min` (`600 s`) timeout.

### 2.4 Mooncake Store, NIXL Trace V2, Dynamo & TE JAX MoE (`e764353f`, `5872ee87`, `56c1c2f7`, `a789babb`, `eb3c860a`)
- **Mooncake Batch Soft-Pin (`e764353f`) & Standby Restore (`5872ee87`)**: Exposes `POST /api/v1/soft_pin` and `POST /api/v1/soft_unpin` (`1..4096` keys/batch) on the admin HTTP port without data transfer, raises `max_kv_soft_pin_ttl` **`7x`** from `86,400,000 ms` (`24 h`) to **`604,800,000 ms` (`7 days`)**, and introduces `PublishRestoredObjects` atomic publication.
- **NIXL Trace Plugin API V2 (`56c1c2f7`)**: Replaces `pushCorrelationId(uint64_t)` with `pushContext(const TraceContext &)` across `postXferReq` and `xfer.complete`, bumping the trace plugin ABI from `V1` to **`V2`**.
- **Dynamo `/health` (`a789babb`) & TE JAX MoE (`eb3c860a`)**: Makes Dynamo `/health` status-only by default (`?instances=true` opt-in avoids tens of milliseconds of discovery serialization per probe), and scopes `NCCL_EP_HT_EM_AG_SCAN_MODE=1` to non-fused split-dispatch tests in TE JAX MoE.

---

## 3. Architect Design Exercises & Falsifiable Questions

1. **Calculation Exercise (SM103 Register Budget)**: Verify that for `threads_per_cta = 512`, the maximum per-thread register allocation at launch is $65536 / 512 = 128$, explaining why `setmaxnreg.dec 136` traps at `q_stage = 1` whereas `q_stage = 2` (`num_regs_other = 96`) succeeds without the cap.
2. **Falsifiable Hypothesis**: Pairing Megatron ModelOpt MXFP4 QAT with vLLM DeepGEMM MoE rollout (dynamic FP8 group-128 activations) reduces training-inference `k3_kl` on `Qwen3.5-35B-A3B` by `5.643x` (`0.00790 -> 0.00140`).
