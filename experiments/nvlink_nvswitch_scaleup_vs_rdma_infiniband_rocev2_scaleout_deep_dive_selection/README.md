# NVLink/NVSwitch (Scale-Up) vs. InfiniBand & RoCEv2 RDMA (Scale-Out): GPUDirect Path, Lossless Tuning & busbw Benchmark

Deterministic quantitative evaluation model for [`docs/components/nvlink-nvswitch-scaleup-vs-rdma-infiniband-rocev2-scaleout-deep-dive-selection-faq.md`](../../docs/components/nvlink-nvswitch-scaleup-vs-rdma-infiniband-rocev2-scaleout-deep-dive-selection-faq.md) and [`.en.md`](../../docs/components/nvlink-nvswitch-scaleup-vs-rdma-infiniband-rocev2-scaleout-deep-dive-selection-faq.en.md).

## Provenance
- **Label**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)

## Run & Verify
```bash
python experiments/nvlink_nvswitch_scaleup_vs_rdma_infiniband_rocev2_scaleout_deep_dive_selection/run.py
python experiments/nvlink_nvswitch_scaleup_vs_rdma_infiniband_rocev2_scaleout_deep_dive_selection/run.py --check
python -m unittest tests/test_nvlink_nvswitch_scaleup_vs_rdma_infiniband_rocev2_scaleout_deep_dive_selection.py
```
