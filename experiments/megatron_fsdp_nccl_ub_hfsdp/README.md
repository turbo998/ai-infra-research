# Megatron-FSDP: `nccl_ub` User-Buffer Registration, `MaxPoolAllocator` Double-Buffer & Hybrid-FSDP (`HFSDP`)

Deterministic stdlib model of `NVIDIA/Megatron-LM` (`commit e592c042d575ad86b9a0b9b20239e4c1a1c1f060`):
- `megatron/core/distributed/fsdp/src/megatron_fsdp/distributed_data_parallel_config.py` & `docs/user-guide/features/megatron_fsdp.md`:
- Quantifies how `HFSDP` (`optim_grads_params` on 8-GPU NVLink + `optim` on 64-node IB) reduces per-GPU state memory by `4.79x` (`244.20 -> 51.01 GiB`) versus `HSDP` (`outer_dp_sharding_strategy='no_shard'`), while `nccl_ub=True` + `MaxPoolAllocator` + Copy-Engine AG (`0/4` SMs vs `16/16` SMs) + `fp8_param_gather=True` speeds up 512-GPU step time by `1.344x` (`572.35 -> 425.95 ms`).
