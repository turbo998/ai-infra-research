# Heterogeneous GPU Cluster Scheduling (`H100` + `B200`) Offline Experiment

Deterministic stdlib evaluation (`experiments/heterogeneous_cluster_scheduler/run.py`) modeling:
1. Megatron-LM custom pipeline layout (`PipelineParallelLayerLayout` DSL) for mixed `H100` (`8` stages) + `B200` (`8` stages) `PP=16, VPP=2` training on a `61`-layer + `1`-MTP DeepSeek-V3-class model.
2. Cross-generation Disaggregated KV Cache RDMA transfer & transcode latency (`SM100 NVFP4 288B` -> `SM90 FP8 656B` vs `BF16 1152B`) across `4K–128K` context lengths.

## Usage

```powershell
python experiments\heterogeneous_cluster_scheduler\run.py
python experiments\heterogeneous_cluster_scheduler\run.py --check
```
