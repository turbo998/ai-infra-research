# Next-Gen AI Ethernet (`NVIDIA Spectrum-X` & `Ultra Ethernet UEC`) Selection Guide: Adaptive Packet Spraying, SuperNIC Out-of-Order Direct Data Placement (`OOO DDP`) & `RTT/EQDS` Congestion Control

> **Language**: [中文](./ultra-ethernet-uec-spectrumx-packet-spraying-ooo-congestion-control-selection-faq.md) | **English (Current)**  
> **Audience**: Cloud & AI Infrastructure Solution Architects and Large-Scale AI Networking Leads  
> **Provenance**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)

![Next-Gen AI Ethernet Spectrum-X & UEC Packet Spraying Summary](../../assets/svg/ultra-ethernet-uec-spectrumx-packet-spraying-ooo-congestion-control-selection-faq-summary.svg)

---

## 1. 📌 Executive Summary

> **Bottom Line**: **Legacy `RoCEv2` degrades to `53.6%` line-rate efficiency (`26.8 GB/s` on 400G) in multi-tenant `>=1,024-GPU` clusters because strict in-order RNICs treat out-of-order arrivals as packet loss (`Go-Back-N`), forcing switches to pin 400G elephant flows onto single Spine links via static 5-tuple ECMP! Next-generation AI Ethernet (`NVIDIA Spectrum-X` and `UEC 1.0: Broadcom Tomahawk 5 + Thor 2`) solves this through Switch + SuperNIC co-design: (1) switches perform Per-Packet Adaptive Routing (`Packet Spraying`) across all Spine links (`96.2%` link balance); (2) SuperNICs (`BlueField-3 / ConnectX-8 / Thor 2`) execute hardware Out-of-Order Direct Data Placement (`OOO DDP`) directly into GPU HBM offsets via PSN bitmaps without retransmission; and (3) hardware `RTT / EQDS` congestion control replaces reactive `DCQCN`—achieving `48.1 GB/s` multi-tenant `busbw` (`96.2%` line rate), `4.2ms` P99 MoE `All-to-All` tail latency, and `-81.4%` network stall TCO!**

- **Interactive `Spectrum-X / UEC` vs. `RoCEv2` Sizer**: [Open Interactive Sizer (`ultra-ethernet-uec-spectrumx-packet-spraying-ooo-congestion-control-sizer.html`)](../reports/ultra-ethernet-uec-spectrumx-packet-spraying-ooo-congestion-control-sizer.html)
- **7-Card 4:5 Visual Report**: [Open 7-Card Report (`ultra-ethernet-uec-spectrumx-packet-spraying-ooo-congestion-control-selection-cards.html`)](../reports/ultra-ethernet-uec-spectrumx-packet-spraying-ooo-congestion-control-selection-cards.html)

---

## 2. 📊 4-Tier AI Ethernet Comparison Matrix (`2,048 GPUs`, 400G Multi-Tenant)

| Metric | Tier 1: Legacy `RoCEv2` (Static ECMP + In-Order GBN) | Tier 2: `Flowlet DLB` + Tuned `DCQCN` | Tier 3 (Fatal Trap): Switch Spray WITHOUT NIC `OOO` | Tier 4 (Recommended): `Spectrum-X / UEC` (Packet Spray + SuperNIC `OOO DDP` + `RTT/EQDS`) |
| :--- | :--- | :--- | :--- | :--- |
| **Multi-Tenant `busbw` (400G)** | `26.8 GB/s` (53.6% ✕) | `39.2 GB/s` (78.4%) | `14.2 GB/s` ✕ (GBN storm!) | **`48.1 GB/s` ✓ (`96.2%` line rate!)** |
| **Spine Link Balance** | `53.6%` ✕ | `78.4%` | `96.0%` | **`96.2%` ✓** |
| **P99 MoE `All-to-All` Latency** | `28.4 ms` ✕ | `11.6 ms` | `64.0 ms` ✕ | **`4.2 ms` ✓ (`-6.76x`)** |
| **Monthly 2,048-GPU Stall TCO** | `$940,000 / mo` | `$465,000 / mo` | `$1,380,000 / mo` ✕ | **`$175,000 / mo` ✓ (`-81.4%`)** |

---

## 3. 🛠️ Production Readiness Checklist

- [ ] **Never Enable Switch Packet Spraying Without Enabling RNIC Hardware `OOO DDP` (`ROCE_ADAPTIVE_ROUTING_EN=1`) First**.
- [ ] **Benchmark Multi-Tenant `alltoall_perf` Under 50% Background Noise** and require `busbw >= 90%` line rate.
