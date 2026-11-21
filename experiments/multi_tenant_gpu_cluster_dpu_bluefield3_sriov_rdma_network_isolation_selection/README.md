# Multi-Tenant GPU Cluster Network Isolation & DPU Offload Experiment (`BlueField-3 DPU` vs. `SR-IOV + Multus` vs. `IB P_Key / RoCEv2 Per-VF QoS`)

Reproducible analytical model comparing 4 multi-tenant RDMA network isolation architectures across a `1,024-GPU` `400GbE / NDR` cluster during a co-located tenant `AlltoAll` incast / PFC pause storm:
- **Tier 1 (`hostNetwork: true` on shared lossless queue)**: `62.0%` victim `busbw` collapse during a noisy-neighbor PFC storm, `18` host CPU cores consumed, zero control-plane isolation.
- **Tier 2 (`Multus CNI + SR-IOV VF + VLAN`)**: `365.0 Gbps` `busbw`, `28.5%` victim `busbw` drop under shared ToR switch buffer exhaustion.
- **Tier 3 (`SR-IOV + IB P_Key / RoCEv2 Per-VF Hardware Pacing + Dynamic Switch Buffers`)**: `376.0 Gbps` `busbw`, `4.2%` victim drop.
- **Tier 4 (`NVIDIA BlueField-3 DPU / SuperNIC Hardware Offload + RTT Telemetry CC + PFC Watchdog`, Recommended)**: `388.0 Gbps` (`97%` line rate), `<0.8%` victim `busbw` jitter, `0` host CPU cores used, and full zero-trust bare-metal isolation.

## Run & Verify

```bash
python experiments/multi_tenant_gpu_cluster_dpu_bluefield3_sriov_rdma_network_isolation_selection/run.py
python experiments/multi_tenant_gpu_cluster_dpu_bluefield3_sriov_rdma_network_isolation_selection/run.py --check
```
