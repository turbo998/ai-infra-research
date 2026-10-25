# FA4 SM100/SM103 2-CTA dQaccum & hd256 Unification, veRL DeepGEMM QAT Rollout, Megatron Mamba Residual State, Mooncake Soft-Pin & NIXL Trace V2 (Phase 85)

Deterministic Python 3.12 standard-library model reproducing the 10 upstream advances across FlashAttention-4, veRL, Megatron-Core, Mooncake Store, NIXL, Dynamo, and TransformerEngine audited in Phase 85 (`priority: 94`):

1. **FlashAttention-4 SM100/SM103 2-CTA Direct Register `dQaccum`, `hd128` `setmaxnreg.dec` Fix & `hd256` Unification (`47e91f1f` PR #2960, `037e808b` PR #2950, `85d6dfe3` PR #2962)**:
   - Models 2-CTA backward postprocess loading `128x128` FP32 `dQaccum` directly from GMEM into registers without staging through SMEM (`128 * 128 * 4 = 65,536 B` SMEM saved vs `32,768 B` BF16 `sdQ`), capping `num_regs_other = min((512 - num_regs_softmax * q_stage - num_regs_correction) // (3 - q_stage) // 8 * 8, 65536 // threads_per_cta)` so `q_stage=1, softmax=176, corr=64` clamps `136 -> 128` on SM103 (`threads_per_cta=512`), and deleting the standalone `2,007`-line `sm100_hd256_2cta_fmha_forward.py` in favor of the unified `flash_fwd_sm100.py` + `NamedBarrier.arrive_and_wait_unaligned` (`aligned=False`).
2. **veRL Unified Megatron ModelOpt QAT & DeepGEMM Low-Precision Rollout Policy (`619c397d` PR #8308 & `8f016846` PR #8287)**:
   - Models the unified trainer-side compressed-tensors export boundary (`FP8_BLOCK`, `MXFP8`, `MXFP4`, `NVFP4`) and DeepGEMM rollout standardization on Blackwell (`Qwen3.5-35B-A3B`, Megatron `EP=8`, vLLM `TP=2`), reproducing measured `k3_kl` divergence reductions with matching QAT:
     - `MXFP8 Linear + MoE`: `0.00387 -> 0.00325` (`1.191x` lower KL)
     - `BF16 Linear + MXFP4 MoE` (DeepGEMM MoE w/ dynamic FP8 group-128 activations): `0.00790 -> 0.00140` (`5.643x` lower KL)
     - `BF16 Linear + NVFP4 MoE` (FlashInfer CUTLASS W4A4): `0.00808 -> 0.00722` (`1.119x` lower KL)
3. **Megatron-Core `ResidualBranchInput` Mamba Local State & Dynamic `dp_cp` Timeout Propagation (`85bf0a66` PR #8344 & `3630e44a` PR #7736)**:
   - Models `ResidualBranchInput[StateT](branch_input, state)` unifying standard `MambaLayer` and `WideResidualMambaLayer` without side-channel attributes, and propagating `distributed_timeout_minutes` (e.g., `45 min = 2700 s` instead of default `10 min = 600 s`) into every dynamic `dp_cp` and `expt_dp` Gloo/NCCL process group.
4. **Mooncake Store Standalone Batch Soft-Pin HTTP Controls & Per-Key Standby Restore (`e764353f` PR #4858 & `5872ee87` PR #4844)**:
   - Models `POST /api/v1/soft_pin` and `POST /api/v1/soft_unpin` supporting `1..4096` keys per batch with `max_kv_soft_pin_ttl` raised `86,400,000 -> 604,800,000 ms` (`24 h -> 7 days`), plus `PublishRestoredObjects` atomic standby restore and removal of the shard-exclusive `kStaleHandleCleanupBatchSize = 64` lock sweep.
5. **NIXL Trace Plugin API V2 `TraceContext` Propagation, Dynamo Opt-In `/health` Discovery & TE JAX MoE FSDP Layout (`56c1c2f7` PR #2445, `a789babb` PR #16454, `eb3c860a` PR #3638)**:
   - Models NIXL V2 `pushContext(const TraceContext &)` carrying full W3C `trace_id` + `span_id` + `sampled` flag across `postXferReq` and `xfer.complete` threads, Dynamo `/health` omitting instance serialization unless `?instances=true` is passed, and TE JAX MoE scoping `NCCL_EP_HT_EM_AG_SCAN_MODE=1` strictly to split-dispatch bitwise layout tests.

## Run

```bash
python experiments/fa4_2cta_verl_deepgemm_mamba_mooncake_nixl_phase85/run.py --check
```
