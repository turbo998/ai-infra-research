# NVLink / NVSwitch (Scale-Up) vs. InfiniBand RDMA vs. RoCEv2 Ethernet RDMA (Scale-Out): Deep Dive, GPUDirect RDMA Zero-Copy, Lossless Tuning & `algbw / busbw` Selection Guide

> **Language**: [中文](./nvlink-nvswitch-scaleup-vs-rdma-infiniband-rocev2-scaleout-deep-dive-selection-faq.md) | **English (Current)**  
> **Audience**: Cloud & AI Infrastructure Solution Architects, AI Networking Architects, and HPC/LLM Platform Leads  
> **Provenance**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)

![NVLink/NVSwitch Scale-Up vs. InfiniBand & RoCEv2 RDMA Scale-Out Summary](../../assets/svg/nvlink-nvswitch-scaleup-vs-rdma-infiniband-rocev2-scaleout-deep-dive-selection-faq-summary.svg)

---

## 1. 📌 Executive Summary

> **Bottom Line**: **AI cluster fabrics are strictly divided into two tiers: Scale-Up (`NVLink 4/5 + NVSwitch`, `900–1,800 GB/s` bidirectional per GPU, hardware load/store memory semantics + `NVLS` in-switch reduction, dedicated to `TP` and intra-node/rack `MoE EP`) and Scale-Out (`InfiniBand RDMA` or `RoCEv2 Ethernet RDMA`, `400G/800G = 50–100 GB/s` per NIC, carrying cross-node `DP/FSDP`, `PP`, cross-node `DeepEP`, and `PD Disaggregated KV Cache` transfers). For Scale-Out selection: choose `InfiniBand NDR/XDR` for `>=1,000-GPU` synchronous frontier pre-training and heavy cross-node MoE (native link-layer credit lossless flow control + `Adaptive Routing` + `SHARPv3`); choose tuned `AI Ethernet RoCEv2` for `<=512–1,024 GPU` training, PD inference pools, and cloud-native fleets—provided you enforce `nvidia-peermem GPUDirect RDMA` zero-copy, `Rail-Optimized topology + NCCL PXN`, and `ECN-before-PFC + DCQCN + DLB/Packet Spraying + 200ms PFC Watchdog`, achieving `45.4–47.6 GB/s` `nccl-tests` `busbw` (`90.8%–95.2%` line rate) and cutting monthly TCO by `68.3%` vs. untuned RoCEv2 (`11.8 GB/s`)!**

- **Interactive `algbw / busbw` & Fabric Sizer**: [Open Interactive Sizer (`nvlink-nvswitch-scaleup-vs-rdma-infiniband-rocev2-scaleout-deep-dive-sizer.html`)](../reports/nvlink-nvswitch-scaleup-vs-rdma-infiniband-rocev2-scaleout-deep-dive-sizer.html)
- **7-Card 4:5 Visual Report**: [Open 7-Card Report (`nvlink-nvswitch-scaleup-vs-rdma-infiniband-rocev2-scaleout-deep-dive-selection-cards.html`)](../reports/nvlink-nvswitch-scaleup-vs-rdma-infiniband-rocev2-scaleout-deep-dive-selection-cards.html)

---

## 2. 🎯 Architect Deep-Dive Q&A

### Q1: How do Scale-Up (`NVLink 4/5 + NVSwitch`) and Scale-Out (`InfiniBand / RoCEv2 RDMA`) differ in bandwidth, semantics, and traffic assignment?
- **Scale-Up (`HGX 8-GPU` or `GB200 NVL72` Rack)**:
  - **Bandwidth & Latency**: `900 GB/s` bidirectional per GPU on NVLink 4 (`H100/H200`) and **`1,800 GB/s`** on NVLink 5 (`B200 / GB200 NVL72`), with **`~0.6 us`** hop latency and hardware `NVLS` in-switch collective reduction.
  - **Traffic**: Dedicated to per-layer synchronous **`TP` (Tensor Parallelism)** and intra-node/rack **`MoE EP`**.
- **Scale-Out (`8x 400G / 800G` NICs across nodes)**:
  - **Bandwidth & Latency**: `50 GB/s` (`400G`) or `100 GB/s` (`800G`) per GPU (`1/18th` of NVLink!), with `1.3–2.5 us` switch hop latency over Queue Pair RDMA Verbs.
  - **Traffic**: Carries **`DP/FSDP` gradient sync**, **`PP` point-to-point activations**, **cross-node `DeepEP` internode dispatch**, and **PD Disaggregated KV Cache transfers**.

### Q2: How does `GPUDirect RDMA` zero-copy work, and why does missing `nvidia-peermem` or leaving `PCIe ACS` enabled drop 400G bandwidth from `47.6 GB/s` to `11.8 GB/s`?
- Without `nvidia-peermem` or with BIOS **PCIe ACS (Access Control Services)** enabled, PCIe Peer-to-Peer (`P2P`) transactions between the GPU and NIC under the same PCIe Gen5 Switch (`PIX`) are blocked and forced to bounce through **Host CPU System DRAM**, cutting `busbw` by **`4x`** (`11.8 GB/s`).
- With `nvidia-peermem` loaded, `ACS` disabled, and `PIX` topology verified via `nvidia-smi topo -m`, the RNIC DMAs directly into/out of GPU `BAR1` HBM with zero CPU/RAM involvement (`45.4–48.2 GB/s`).

### Q3: How do `InfiniBand` and `RoCEv2` differ, and how do we prevent ECMP hash collisions and PFC pause storms on RoCEv2?
1. **Flow Control**: InfiniBand uses native **link-layer credit-based flow control** (zero congestion drops or pause deadlocks by design). RoCEv2 runs over UDP/IP (`port 4791`) and must configure **early `ECN` marking ($K_{\min} \ll \text{PFC XOFF}$) + `DCQCN` rate control + `PFC` on Priority Queue 3 + `200ms PFC Watchdog`**.
2. **Topology & Load Balancing**: Use a **Rail-Optimized Clos topology** (connecting all `GPU #i` to `Rail Leaf #i`) paired with **`NCCL PXN`** (routing cross-rail messages over intra-node `900 GB/s NVLink` first so they exit on the target Rail Leaf in 1 hop) plus **Flowlet `DLB` or `Packet Spraying`** (`Spectrum-X / UEC`) to eliminate static 5-tuple ECMP elephant-flow collisions.

### Q4: How do we convert between `algbw` and `busbw` in `nccl-tests`?
- **Algorithm Bandwidth (`algbw`)**: $\text{algbw} = S / t$.
- **Bus Bandwidth (`busbw`, true hardware wire utilization)**:
  - **Ring `All-Reduce`**: $\text{busbw} = \text{algbw} \times \frac{2(n - 1)}{n}$
  - **`All-Gather` / `Reduce-Scatter` / `All-to-All`**: $\text{busbw} = \text{algbw} \times \frac{n - 1}{n}$
- On a 512-GPU 400G (`50 GB/s`) cluster running a `4 GB` `All-Reduce` in `167.8 ms`: $\text{algbw} = 23.84\text{ GB/s}$ and $\text{busbw} = 23.84 \times \frac{1022}{512} = \mathbf{47.59\text{ GB/s}}$ (`95.2%` line-rate efficiency).

---

## 3. 🧭 Visual Diagrams

### 3.1 Scale-Up vs. Scale-Out Fabric Selection Decision Tree
```mermaid
flowchart TD
    A["AI Cluster Network Design<br/>(Scale-Up + Scale-Out)"] --> B{"Traffic Layer?"}
    B -- "Per-Layer Sync TP /<br/>Intra-Node or Rack MoE EP" --> C["Scale-Up Fabric:<br/>NVLink 4 (900 GB/s) / NVLink 5 (1,800 GB/s)<br/>+ NVSwitch (NVLS Hardware Reduction)"]
    B -- "Cross-Node DP/FSDP / PP /<br/>Cross-Node DeepEP / PD KV Transfer" --> D{"Cluster Scale & Ops Profile?"}
    D -- ">= 1,000-GPU Sync Pre-Training /<br/>Heavy Cross-Node MoE / Turnkey Lossless" --> E["Option A: InfiniBand NDR 400G / XDR 800G<br/>• Native Credit Lossless (~1.3us)<br/>• Adaptive Routing + SHARPv3"]
    D -- "<= 512-1,024 GPU Training / PD Serving /<br/>Cloud-Native Ethernet Standard" --> F["Option B: Tuned AI Ethernet RoCEv2<br/>• Rail-Optimized + NCCL PXN + GPUDirect<br/>• ECN-before-PFC + DCQCN + DLB/Spray + Watchdog"]
```

---

## 4. 📊 4-Tier Fabric Comparison Matrix (512-GPU 400G Cluster)

| Metric | Tier 1: Untuned RoCEv2 (No Peermem / Static ECMP) | Tier 2: Tuned AI Ethernet RoCEv2 (`DCQCN + DLB`) | Tier 3: `InfiniBand NDR 400G` (`AR + SHARPv3`) | Tier 4 (Recommended): `NVLink/NVLS` + `Rail RDMA + PXN` |
| :--- | :--- | :--- | :--- | :--- |
| **`nccl-tests` AllReduce `busbw`** | `11.8 GB/s` (23.6% ✕) | `45.4 GB/s` (90.8% ✓) | **`48.2 GB/s` (96.4% ✓)** | **`47.6 GB/s` (95.2% ✓)** |
| **P99 Collective Tail Latency** | `94.0 ms` ✕ | `8.2 ms` ✓ | `5.4 ms` ✓ | **`4.9 ms` ✓** |
| **Monthly PFC Deadlocks** | `14 / mo` ✕ | `0 / mo` ✓ | `0 / mo` ✓ | **`0 / mo` ✓** |
| **Monthly 512-GPU Network & Stall TCO** | `$1,150,000 / mo` | `$410,000 / mo` | `$465,000 / mo` | **`$365,000 / mo` (`-68.3%`)** |

---

## 5. 🛠️ Production Readiness Checklist

- [ ] **GPUDirect RDMA Gate**: Verify `nvidia_peermem` is loaded, BIOS `PCIe ACS` is disabled, and `nvidia-smi topo -m` reports `PIX` for every `(GPU_i, NIC_i)` pair.
- [ ] **Rail-Optimized + `NCCL PXN`**: Enable `NCCL_PXN_DISABLE=0` and `NCCL_NVLS_ENABLE=1` to eliminate cross-rail Spine hops.
- [ ] **RoCEv2 Lossless Tuning (`ECN < PFC + DCQCN + Watchdog`)**: Set ECN $K_{\min}$ well below `PFC XOFF` and enforce a `200ms PFC Watchdog` on all ToRs.
- [ ] **Acceptance Gate (`busbw >= 90%` Line Rate)**: Require `>=45.0 GB/s` `busbw` on 400G NICs (`>=90.0 GB/s` on 800G) in `nccl-tests` `all_reduce_perf` before production release.
