# 🏆 Milestone `#625`: Enterprise AI High-Speed Fabric Bible — Scale-Up (`NVLink 4/5 · NVSwitch · GB200 NVL72`) vs. Scale-Out (`InfiniBand NDR/XDR` vs. `RoCEv2 Ethernet RDMA`), `GPUDirect RDMA` Zero-Copy Path, Lossless `PFC+ECN+DCQCN` Buffer Math, `Rail-Optimized` Topology & `algbw/busbw` Conversion Guide (`#625`)

> **Language / 语言**: [简体中文](./enterprise-milestone-625-nvlink-rdma-ib-rocev2-lossless-busbw-sizing-faq.md) | **English** · [📱 Interactive H5 Study Lab (with SVG Zero-Copy Path + PFC/busbw Calculator)](../h5/milestone-625-nvlink-rdma-ib-rocev2-lossless-busbw-lab.html) · [🧮 Interactive Sizer](../reports/enterprise-milestone-625-nvlink-rdma-ib-rocev2-lossless-busbw-sizing-sizer.html) · [🃏 7-Card Visual Report](../reports/enterprise-milestone-625-nvlink-rdma-ib-rocev2-lossless-busbw-sizing-cards.html) · [Back to Catalog](../full-catalog-en.md)

![Architecture Summary](../../assets/svg/enterprise-milestone-625-nvlink-rdma-ib-rocev2-lossless-busbw-sizing-faq-summary.svg)

---

## 1. 3-Minute Executive Summary

1. **Bottom-Line Recommendation**:
   - AI fabrics operate across two distinct physical domains:
     - **Scale-Up (`NVLink 4/5 + NVSwitch / GB200 NVL72`)**: Delivers **`900 GB/s` (H100/H200) to `1,800 GB/s` (B200/GB200)** bidirectional bandwidth per GPU at **`~0.5–0.6 μs`** memory-semantic latency (`14×–28×` faster than `PCIe Gen5 x16` at `64 GB/s`), purpose-built for synchronous **Tensor Parallelism (`TP`) and rack-scale `EP=72`**.
     - **Scale-Out (`InfiniBand NDR/XDR` vs. `RoCEv2 / Spectrum-X / UEC`)**: Delivers **`50–100 GB/s`** unidirectional bandwidth per 400G/800G NIC at **`1.5–3.5 μs`** latency (`18×–36×` slower than NVLink), purpose-built for **Data Parallelism (`DP/FSDP`), Pipeline Parallelism (`PP`), inter-node `DeepEP All-to-All`, and PD Disaggregation `KV Cache` transfers**.
   - On a 512-GPU (`64 × 8-GPU`) cluster, leaving PCIe **`ACS=On`** (breaking `GPUDirect RDMA` into CPU bounce buffers) or running RoCEv2 with **naive `PFC` without early `ECN/DCQCN` marking and `DLB/Packet Spraying`** collapses 400G `AllReduce busbw` from `50 GB/s` to **`14.2–26.5 GB/s`** and causes a **`12.5%` monthly PFC pause-storm deadlock rate**.
   - Enforcing **`GPUDirect RDMA` (`PIX` single-root switch path) + `Rail-Optimized + NCCL PXN` topology + early `ECN (150 KB)` / `DCQCN` + `200ms PFC Watchdog` + `DLB/Packet Spraying`** achieves **`47.8 GB/s busbw` (`95.6%` line rate, `3.37×` faster)**, eliminates PFC deadlocks (**`0.0%`**), and cuts optical transceiver CapEx by **`56.2%` (`-$1.42M`)** vs. a 3-tier Fat-Tree!

---

## 2. Key Architecture Formulas: PFC Headroom & `algbw` ↔ `busbw`

1. **Lossless Switch `PFC Headroom` Buffer Sizing Formula**:
   $$B_{\text{headroom}} \ge 2 \times \left(T_{\text{cable}} + T_{\text{phy/mac}}\right) \times \text{BW}_{\text{port}} + 2 \times \text{MTU}$$
   - For a `400GbE` (`50 GB/s`) port over `100m` fiber ($500\text{ ns}$ one-way) + `900 ns` PHY/MAC delay and `4,224 B` MTU:
     $$B_{\text{headroom}} \ge 2 \times 1.4\ \mu\text{s} \times 50\text{ B/ns} + 8,448\text{ B} \approx \mathbf{148.5\text{ KB/port}} \quad (\text{Provision } \mathbf{192\text{--}256\text{ KB}})$$
2. **`nccl-tests` `algbw` to `busbw` Conversion Formulas ($N$ GPUs)**:
   - **`AllReduce`**: $\text{busbw} = \text{algbw} \times \frac{2(N - 1)}{N}$
   - **`AllGather / ReduceScatter / All-to-All`**: $\text{busbw} = \text{algbw} \times \frac{N - 1}{N}$

---

## 3. Quantitative 512-GPU Fabric Comparison Matrix (`8 × 400G Multi-Rail`)

| Metric | Option A: Untuned Cloud RoCEv2 (`ACS=On + Static ECMP + Naive PFC`) | Option B: InfiniBand NDR 400G (`Credit Flow + AR + SHARP`) | Option C: Tuned AI Ethernet RoCEv2 (`PIX + Rail/PXN + ECN/DCQCN + DLB`) 🏆 |
| :--- | :--- | :--- | :--- |
| **Inter-Node `AllReduce busbw`** | `14.2 – 26.5 GB/s` (`28.4%–53.0%` ✕) | **`48.6 GB/s` (`97.2%` ✓)** | **`47.8 GB/s` (`95.6%` line rate, `3.37×` faster ✓)** |
| **MoE `All-to-All` P99 Latency (1MB)** | `1,480 μs` (ECMP collision ✕) | **`215 μs`** | **`238 μs` (`6.2×` lower tail latency ✓)** |
| **Monthly PFC Deadlock Rate** | `12.5%` (pause storm ✕) | `0.0%` | **`0.0%` (`ECN 150KB` + `200ms Watchdog` ✓)** |
| **512-GPU Switch + Optics CapEx** | `$2.53M` (3-Tier Fat-Tree) | `$3.45M` (IB premium) | **`$1.68M` (`-56.2%` optics via `Rail+PXN`, saves `$1.77M` vs. IB ✓)** |

---

## 4. Day-0 Pre-Production Verification Checklist

- [ ] **1. `PIX` Topology & `nvidia-peermem`**: Verify `nvidia-smi topo -m` shows `PIX` for every `(GPU_i, mlx5_i)` pair and BIOS `ACS` is disabled.
- [ ] **2. `ECN < PFC` Watermarks & Watchdog**: Configure `K_min = 150 KB`, `K_max = 1,500 KB`, `PFC Headroom >= 192 KB`, and hardware `PFC Watchdog = 200 ms`.
- [ ] **3. `Rail-Optimized + NCCL PXN`**: Enable `NCCL_PXN_DISABLE=0` so cross-rail collectives route across intra-node NVLink before entering the target rail leaf.
- [ ] **4. `>= 90%` Line-Rate `busbw` Gate**: Require `busbw >= 45.0 GB/s` on 400G NICs across `all_reduce_perf` and `alltoall_perf` before accepting cluster handover.
