# GPU Container Checkpoint/Restore (`CRIU + NVIDIA cuda-checkpoint`), `vLLM Sleep Mode` & Live Spot Migration Selection Experiment

Reproducible wakeup latency, 30-second Spot preemption evacuation success rate, and monthly TCO model comparing 4 GPU pod lifecycle architectures across `20` intermittent `70B-FP8` enterprise models.

## Run
```bash
python experiments/llm_checkpoint_restart_criu_cuda_checkpoint_live_pod_migration_selection/run.py
python experiments/llm_checkpoint_restart_criu_cuda_checkpoint_live_pod_migration_selection/run.py --check
```
