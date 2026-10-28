# Phase 125 Track C: Megatron-LM TP Shared Expert, TorchTitan Configurable Compile & Loss Redesign, verl TPU v7x GRPO & Submission Gate, Mooncake EFA/NUMA/HCCS, LMCache & Dynamo

Deterministic CPU/offline analytical suite (`own-offline`) modeling:
1. **Megatron-LM TP Fused Shared Expert FC1 & TorchTitan Compile/Loss Redesign (`#8145`, `3e47f31e`; `#5463`, `0f9fd2bd`; `#4582`, `740d5f75`; `#5569`, `36af428d`)**: Eliminates TP>1 shared-expert FC1 shape mismatch, enables out-of-tree `LocalCompileConfig` backends (`1.314x`), and saves `62.5%` activation memory via decoupled `ChunkedLossWrapper` + `MTPLoss`.
2. **verl TPU v7x Qwen3.5-35B-A3B GRPO, vLLM Submission-Gate Barrier & VeOmni ROCm/FSDP2 (`#8382`, `2ab35292`; `#8397`, `081c76bb`; `#8371`, `6d04f398`; `#8370`, `b65da71c`; `#8369`, `6f3f449e`)**: `260.5x` faster pre-weight-sync submission gate drain (`4820.0 ms -> 18.5 ms`), `2.38x` FSDP2 CPU-offload fused AdamW speedup, and `3.65x` EP-sharded checkpoint load speedup.
3. **Mooncake HCCS/TENT/EFA/NUMA, LMCache Lazy SDK & Dynamo Affinity/Router (`#5049`, `#4938`, `#4888`, `#4956`, `#4915`, `#5944`, `#6134`, `#6090`, `#16969`, `#16721`, `#17474`, `#17325`)**: `0.0` EFA false peer aborts via `MC_EFA_STALL_MIN_MS=1000`, `0` NUMA `mbind(EINVAL)` errors under `cpuset.mems`, and `0.0` Dynamo fallback worker stale DP-rank misroutes.

## Run
```bash
python experiments/megatron_tp_shared_exp_torchtitan_compile_loss_verl_tpu_grpo_gate_rocm_mooncake_lmcache_dynamo_phase125/run.py
python experiments/megatron_tp_shared_exp_torchtitan_compile_loss_verl_tpu_grpo_gate_rocm_mooncake_lmcache_dynamo_phase125/run.py --check
```
