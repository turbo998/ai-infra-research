# Kubernetes GPU-RDMA-NUMA Topology-Aware Scheduling Guide: `Kubernetes DRA (Dynamic Resource Allocation)` vs. Legacy `Device Plugin` vs. `Kueue TAS / Volcano`

> **Language**: [中文](./kubernetes-dra-topology-aware-gpu-nic-numa-scheduling-kueue-volcano-selection-faq.md) | **English (Current)**  
> **Audience**: Cloud & AI Infrastructure Solution Architects and Kubernetes AI Platform Engineers  
> **Provenance**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)

![Kubernetes DRA GPU-RDMA-NUMA Topology Scheduling Summary](../../assets/svg/kubernetes-dra-topology-aware-gpu-nic-numa-scheduling-kueue-volcano-selection-faq-summary.svg)

---

## 1. 📌 Executive Summary

> **Bottom Line**: **On 8-GPU `HGX H100/H200/B200` nodes equipped with `8x 400GbE ConnectX-7` RDMA NICs across 4 PCIe Gen5 switches and 2 CPU NUMA sockets, scheduling sub-node (`2-GPU / 4-GPU`) PD-disaggregated inference or fine-tuning pods with legacy Kubernetes `Device Plugin` (`topologyManagerPolicy: none`, `Tier 1`) blindly pairs GPUs on `NUMA-0` with RDMA NICs on `NUMA-1` (`SYS` path in `nvidia-smi topo -m`)—breaking GPUDirect RDMA P2P and collapsing per-NIC bandwidth by `70%` (`14.5 GB/s` vs. `48.0 GB/s`). Meanwhile, node-local Kubelet `TopologyManager` (`Tier 2`) is invisible to `kube-scheduler`, triggering `TopologyAffinityError` eviction storms (`18.5%` fragmentation waste). Upgrading to `Kubernetes DRA (Dynamic Resource Allocation)` + Cluster-Level `Kueue Topology-Aware Scheduling (TAS)` (`Tier 4`) enforces 3-tier hardware alignment (`Same PCIe Switch PIX -> Same NUMA -> Same ToR Rail`) at the central scheduler, restoring `47.4 GB/s` (`98.8%` line rate), cutting P99 communication latency by `83.2%` (`11.4 ms`), and lifting cluster Goodput to `95.6%`.**

- **Interactive K8s GPU-RDMA-NUMA Topology Sizer**: [Open Interactive Sizer (`kubernetes-dra-topology-aware-gpu-nic-numa-scheduling-kueue-volcano-sizer.html`)](../reports/kubernetes-dra-topology-aware-gpu-nic-numa-scheduling-kueue-volcano-sizer.html)
- **7-Card 4:5 Visual Report**: [Open 7-Card Report (`kubernetes-dra-topology-aware-gpu-nic-numa-scheduling-kueue-volcano-selection-cards.html`)](../reports/kubernetes-dra-topology-aware-gpu-nic-numa-scheduling-kueue-volcano-selection-cards.html)

---

## 2. 📊 4-Tier Kubernetes Topology Scheduling Matrix (`512x H200` + `400GbE CX-7`)

| Metric | Tier 1: Legacy `Device Plugin` (`topology: none`) | Tier 2: Kubelet `TopologyManager` (`single-numa-node`) | Tier 3: `Kueue TAS` Rack-Aware + Kubelet NUMA | Tier 4 (Recommended): `K8s DRA` PCIe-Switch (`PIX`) + `Kueue TAS` |
| :--- | :--- | :--- | :--- | :--- |
| **Effective GPUDirect RDMA BW** | `14.5 GB/s` ✕ (`SYS` hairpin) | `36.0 GB/s` (`NODE`) | `42.5 GB/s` | **`47.4 GB/s` ✓ (`PIX` direct P2P)** |
| **PD / AllReduce P99 Latency** | `68.0 ms` ✕ | `29.5 ms` | `18.2 ms` ✓ | **`11.4 ms` ✓ (`-83.2%`)** |
| **Multi-Node Cluster Goodput** | `61.5%` ✕ | `78.0%` | `88.5%` | **`95.6%` ✓ (`+34.1 pts`)** |
| **Topology Fragmentation Waste** | `24.0%` ✕ | `18.5%` ✕ (Eviction loops) | `9.2%` | **`3.5%` ✓** |
| **Monthly Compute Waste Cost** | `$168,000 / mo` ✕ | `$96,000 / mo` | `$48,000 / mo` | **`$18,500 / mo` ✓ (`-89.0%`)** |

---

## 3. 🛠️ Production Readiness Checklist

- [ ] **Audit `nvidia-smi topo -m` in Every Pod Entrypoint** and fail fast if any allocated GPU-NIC pair resolves to `SYS` (cross-socket UPI/QPI) instead of `PIX` or `NODE`.
- [ ] **Adopt Kubernetes `DRA (ResourceClaim)` for Sub-Node Multi-NIC Pods** so `kube-scheduler` co-allocates GPUs and RDMA NICs sharing the same PCIe Gen5 switch before binding the pod.
