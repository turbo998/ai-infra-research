# RL Post-Training Weight Resharding, CuMem Sleep/Wake & Long-CoT Straggler Model

- **Measurement type**: `own-offline` (deterministic CPU analytical model derived from `volcengine/verl` and `vllm-project/vllm` source code).
- **Clear Boundary**: This experiment does not launch Ray workers or multi-node GPU training clusters.

## What It Models

1. **671B MoE Weight Resharding & Transfer (`verl/checkpoint_engine/*`, `bucketed_weight_transfer.py`)**:
   - Compares colocated `BucketedWeightSender` CUDA IPC (`512 MiB` bucket, `0.0835 s`), shared-memory fallback (`use_shm=True`, `1.7954 s`), disaggregated single-sender NCCL broadcast (`31.8222 s`), 8-sender NVLink relay NCCL broadcast (`multi_sender=True`, `3.9778 s`), and `DeltaShardedCheckpointEngine` sparse `(int32 idx, bf16 val)` delta flush (`2.0620 s`, `1120.0 MiB` peak receiver staging).
2. **vLLM `CuMemAllocator` Sleep/Wake Lifecycle (`vllm/device_allocator/cumem.py`)**:
   - Compares co-resident `No Sleep` (`106.0312 GiB` peak HBM -> OOM on `80 GiB` GPU), `Sleep Level 1` (`offload_tags=("weights",)`, `58.5 GiB` peak HBM, `0.9118 s` PCIe D2H+H2D transition), and `Sleep Level 2` (unmap weights + KV cache, `58.5 GiB` peak HBM, `0.038 s` VM unmap/remap overhead before direct IPC weight refit).
3. **Long-CoT GRPO Straggler Bubble**:
   - Quantifies GPU idle bubble under synchronous step barrier (`77.73%` idle waiting for `32K` responses) vs partial rollout checkpoint recycle (`61.72%` bubble, `2.0x` speedup) and staleness-bounded continuous async rollout (`8.5%` bubble, `4.1095x` speedup).

## Usage

```bash
python experiments/rl_rollout_resharding/run.py
python experiments/rl_rollout_resharding/run.py --check
```
