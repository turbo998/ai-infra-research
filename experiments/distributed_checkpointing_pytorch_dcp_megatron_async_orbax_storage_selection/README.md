# Distributed Training Checkpointing Selection Experiment (`PyTorch Async DCP` vs. `Megatron-Core DistCkpt` vs. `JAX Orbax`)

Reproducible analytical model comparing 4 distributed training checkpointing architectures across a `1,024-GPU` cluster saving a `6.4 TB` training state every `30 minutes`:
- **Tier 1 (`torch.save` on `rank==0`)**: `480.0 sec` GPU blocking stall, `76.9%` checkpoint Goodput, no topology resharding.
- **Tier 2 (`Synchronous Sharded PyTorch DCP / Orbax`)**: `38.0 sec` stall, `97.9%` Goodput, full elastic topology resharding.
- **Tier 3 (`Two-Stage Pinned-RAM D2H + Async Background Flush`)**: `1.8 sec` GPU stall, `99.90%` Goodput.
- **Tier 4 (`Zero-Overhead CUDA Stream D2H + NVMe Burst Buffer + Async DCP/Orbax`, Recommended)**: `0.25 sec` GPU stall (`1,920x` faster than Tier 1), `99.98%` Goodput, saving `$333,710/month` in wasted GPU time.

## Run & Verify

```bash
python experiments/distributed_checkpointing_pytorch_dcp_megatron_async_orbax_storage_selection/run.py
python experiments/distributed_checkpointing_pytorch_dcp_megatron_async_orbax_storage_selection/run.py --check
```
