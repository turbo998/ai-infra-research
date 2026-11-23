# Milestone `#510`: Enterprise AI Fabric Bible (`NVLink` vs. `InfiniBand / RoCEv2`, `GPUDirect RDMA`, `PFC+ECN+DCQCN` & `algbw/busbw` Sizer)

Reproducible analytical model for **Scale-Up (`NVLink 4/5 & NVSwitch`) vs. Scale-Out (`InfiniBand NDR/XDR` vs. `RoCEv2 AI Ethernet`)**, **GPUDirect RDMA zero-copy DMA (`48.2 GB/s` vs. `11.4 GB/s` CPU bounce)**, **lossless `PFC + ECN + DCQCN` tuning**, **Rail-Optimized vs. Fat-Tree topologies**, and **NCCL `algbw` to `busbw` conversion**.

## Run

```bash
python experiments/enterprise_milestone_510_nvlink_infiniband_rocev2_gpudirect_busbw_bible/run.py
```

- **Provenance**: `synthetic + anonymized_reported (independently_reproduced=false)`
- **Bilingual Guides**:
  - [Chinese FAQ](../../docs/components/enterprise-milestone-510-nvlink-infiniband-rocev2-gpudirect-busbw-bible-faq.md)
  - [English FAQ](../../docs/components/enterprise-milestone-510-nvlink-infiniband-rocev2-gpudirect-busbw-bible-faq.en.md)
- **Interactive H5 Lab**: [docs/h5/milestone-510-nvlink-ib-rocev2-gpudirect-busbw-lab.html](../../docs/h5/milestone-510-nvlink-ib-rocev2-gpudirect-busbw-lab.html)
