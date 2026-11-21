# Runtime Weight Hot-Reload & RL Actor-to-Rollout Weight Sync Experiment (`CUDA IPC` & `Bucketed FP8 NCCL Broadcast` vs. `Disk Staging`)

Reproducible analytical model comparing 4 runtime weight update architectures across a `128-GPU` `70B` RL/serving cluster (`140 GB` BF16 Actor -> `70 GB` FP8 `vLLM` Rollout, `30.0s` rollout+training compute per step):
- **Tier 1 (`Shared Filesystem .safetensors Dump + vLLM Reload`)**: `95.0 sec` weight sync latency (`24.0%` RL step Goodput, re-captures CUDA Graphs).
- **Tier 2 (`TCP/gRPC Host-Staged In-Place Parameter Update`)**: `22.0 sec` sync latency (`57.7%` Goodput, preserves CUDA Graphs).
- **Tier 3 (`Disaggregated 400G RDMA NCCL Broadcast` in BF16)**: `2.4 sec` sync latency (`92.6%` Goodput).
- **Tier 4 (`Zero-Copy CUDA IPC Memory Handles` [`0.45 sec` co-located] / `Bucketed On-the-Fly FP8 Quantized NCCL Broadcast` [`1.15 sec` disaggregated], Recommended)**: **`0.45 sec` sync latency (`211x` faster than disk staging)**, `100%` in-place CUDA Graph preservation, and **`98.5%` RL step Goodput** (`-$301,950/month` saved).

## Run & Verify

```bash
python experiments/llm_runtime_weight_hot_reload_rl_nccl_broadcast_cuda_ipc_sync_selection/run.py
python experiments/llm_runtime_weight_hot_reload_rl_nccl_broadcast_cuda_ipc_sync_selection/run.py --check
```
