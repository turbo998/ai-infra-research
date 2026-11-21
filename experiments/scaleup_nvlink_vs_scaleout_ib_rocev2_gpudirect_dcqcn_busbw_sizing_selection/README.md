# Experiment: Scale-Up (NVLink) vs. Scale-Out (IB / RoCEv2), GPUDirect RDMA, DCQCN Buffer Math & algbw/busbw Conversion (`#398`)

Reproducible analytical model evaluating four AI fabric configurations across a 512-GPU 400G cluster (`algbw` to `busbw` conversion, PCIe ACS P2P path, ECN/PFC watermarks, and adaptive routing/packet spraying).

- **Provenance**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)
- **Run**: `python experiments/scaleup_nvlink_vs_scaleout_ib_rocev2_gpudirect_dcqcn_busbw_sizing_selection/run.py`
- **Verify**: `python experiments/scaleup_nvlink_vs_scaleout_ib_rocev2_gpudirect_dcqcn_busbw_sizing_selection/run.py --check`
