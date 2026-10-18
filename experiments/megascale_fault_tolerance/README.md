# Mega-Scale (4K–64K GPU) Fault-Tolerant Training: NVRx In-Process Hot Restart, Peer-RAM Checkpointing & SDC Verification

## Overview

This deterministic Python standard-library experiment models the **Effective Training Time Ratio (`ETTR`)** across cluster scales from `4,096` to `65,536` GPUs (`per_gpu_mtbf_hours = 12,000 h`, giving a `65,536`-GPU cluster MTBF of `10.99 minutes`) under three recovery architectures, plus **Silent Data Corruption (`SDC`)** bit-flip verification overhead:

1. **Slurm Full-Job Restart + PFS Checkpoint (`slurm_pfs`)**: Traditional job teardown, `srun`/`torchrun` reallocation, CUDA/NCCL cold initialization, and Parallel File System checkpoint reload (`915–960 s` restart penalty; `95 s+` blocking checkpoint overhead).
2. **TorchElastic Cold Process Restart + Local NVMe (`elastic_local_nvme`)**: Elastic worker respawn with local NVMe checkpoint staging (`135 s` restart penalty; `8.0 s` blocking overhead).
3. **NVIDIA Resiliency Extension (`NVRx`) In-Process Hot Standby + Peer CPU-RAM Replication (`nvrx_inprocess_peer_ram`)**: Uses `nvidia_resiliency_ext.inprocess.Wrapper` with pre-initialized hot standby ranks (`reserve_fn` + `ShiftRanks`), `AbortTorchDistributed`, and `LocalCheckpointManager` (`CliqueReplicationStrategy` over peer CPU RAM / RDMA), cutting restart + restore latency to `19.8 s` (`18.0 s` in-process communicator reconfigure + `1.8 s` peer-RAM state restore) and blocking snapshot overhead to `1.2 s`.

## Reproduction

```bash
python experiments/megascale_fault_tolerance/run.py
python experiments/megascale_fault_tolerance/run.py --check
```

## Key Findings (from `results.json`)

- **64K-GPU ETTR Collapse vs NVRx Recovery**: At `65,536` GPUs (`MTBF = 10.99 min`), traditional Slurm + PFS restart (`960 s`) collapses `ETTR` to **`18.58%`**, whereas NVRx `inprocess.Wrapper` + peer-RAM checkpointing (`19.8 s` recovery, `40 s` optimal checkpoint cadence) sustains **`90.30%` ETTR** (`+71.72 percentage points` gain, `4.86x` effective cluster throughput).
- **16K-GPU Production Regime**: At `16,384` GPUs (`MTBF = 43.95 min`), NVRx improves `ETTR` from `54.86%` (Slurm PFS) and `83.69%` (Elastic NVMe) to **`95.01%`** (`+40.15 pts`).
- **Silent Data Corruption (`SDC`) Orthogonal Checksum Verification**: Flipping any single bit of a finite `BF16` activation/gradient (`0.375`, `exp=125`) produces `16/16` finite non-NaN corruptions (including `exp` bit-14 flips to `~1.5e18` that remain below `exp=255`), so NaN/Inf watchdogs catch **`0.0%`** immediately and suffer `~250`-step delayed rollback, whereas orthogonal FP32 ABFT/gradient checksums catch **`100.0%`** within `1` step at **`0.0732%`** FLOP overhead (`4096^3` GEMM).
