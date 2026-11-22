# Enterprise AI Interconnect Fabric (`NVLink / NVSwitch` vs. `InfiniBand NDR/XDR` vs. `AI Ethernet RoCEv2`), `GPUDirect RDMA` & `algbw/busbw` Sizing Model (`#446`)

Reproducible analytical model comparing PCIe ACS CPU-bounce fallback, static ECMP RoCEv2, Rail-Optimized AI Ethernet RoCEv2 (`GPUDirect + ECN/DCQCN + DLB/Packet Spraying`), and InfiniBand NDR/XDR (`Adaptive Routing + SHARPv3`).

- **Provenance**: `synthetic + anonymized_reported (independently_reproduced=false)`
- **Run**: `python experiments/enterprise_ai_network_nvlink_infiniband_rocev2_gpudirect_busbw_sizing_selection/run.py --check`
