# MoE Training & Serving Full-Stack Architecture: `DeepEP` Dual-Mode Kernels (`Normal` vs. `IBGDA`), `DualPipe / Two-Batch Overlap`, Auxiliary-Loss-Free Routing & `EPLB` Guide

> **Language**: [中文](./moe-architecture-deepep-ibgda-dualpipe-eplb-load-balancing-selection-faq.md) | **English (Current)**  
> **Audience**: Cloud & AI Infrastructure Solution Architects and Large-Scale MoE Platform Leads  
> **Provenance**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)

![MoE Systems Architecture DeepEP IBGDA, Dual-Batch Overlap & EPLB Summary](../../assets/svg/moe-architecture-deepep-ibgda-dualpipe-eplb-load-balancing-selection-faq-summary.svg)

---

## 1. 📌 Executive Summary

> **Bottom Line**: **Large MoE models (`DeepSeek-V3/R1 671B`, `Qwen3-235B-A22B`) trade dense FLOPs for two per-layer cross-node `All-to-All` collectives (`Dispatch` + `Combine`, `~169 KB/token/layer`) and hot-expert straggler skew. Running MoE over generic serial `NCCL All-to-All` wastes `54.8%` of step time waiting on network transfers! Production MoE infrastructure requires four pillars: (1) dual-mode `DeepEP` kernels (`Normal` NVLink-to-RDMA forwarding for Prefill/Training; `Low-Latency IBGDA` GPU-direct NIC doorbell ringing for Decode at `168us`); (2) `DualPipe / Two-Batch Ping-Pong Overlap` (hiding `Batch A` RDMA transfers behind `Batch B` `MLA/FFN` compute, reducing exposed comm stall to `1.8%`); (3) `Auxiliary-Loss-Free Dynamic Bias` routing in training; and (4) hierarchical `EPLB` redundant hot-expert replication in serving (flattening max-to-mean expert load skew from `3.85x` to `1.09x`)—delivering `2,380 Goodput tok/s/GPU` (`4.58x` vs. Tier 1) and cutting `$/1M` tokens by `78.8%` (`$0.25/1M`)!**

- **Interactive MoE `DeepEP` & `EPLB` Sizer**: [Open Interactive Sizer (`moe-architecture-deepep-ibgda-dualpipe-eplb-load-balancing-sizer.html`)](../reports/moe-architecture-deepep-ibgda-dualpipe-eplb-load-balancing-sizer.html)
- **7-Card 4:5 Visual Report**: [Open 7-Card Report (`moe-architecture-deepep-ibgda-dualpipe-eplb-load-balancing-selection-cards.html`)](../reports/moe-architecture-deepep-ibgda-dualpipe-eplb-load-balancing-selection-cards.html)

---

## 2. 📊 4-Tier MoE Serving Comparison Matrix (`64x H200`, `EP=64`, `DeepSeek-R1 671B`)

| Metric | Tier 1: Generic `NCCL All-to-All` (Serial, Static) | Tier 2: `DeepEP` Kernels Only (No Overlap/EPLB) | Tier 3: `DeepEP` + Two-Batch Overlap (No EPLB) | Tier 4 (Recommended): `DeepEP IBGDA` + Two-Batch Overlap + `EPLB` |
| :--- | :--- | :--- | :--- | :--- |
| **Exposed Comm Stall Share** | `54.8%` ✕ | `24.6%` | `6.8%` | **`1.8%` ✓** |
| **Decode `All-to-All` Latency** | `820 us` ✕ | `195 us` | `182 us` | **`168 us` ✓ (`IBGDA`)** |
| **Hot-Expert Skew (`Max/Mean`)** | `3.85x` ✕ | `3.40x` ✕ | `2.95x` ✕ | **`1.09x` ✓ (`EPLB`)** |
| **SLA-Compliant Goodput / GPU** | `520 tok/s` | `1,180 tok/s` | `1,760 tok/s` | **`2,380 tok/s` ✓ (`4.58x`)** |
| **Cost per 1M Tokens** | `$1.18 / 1M` | `$0.52 / 1M` | `$0.35 / 1M` | **`$0.25 / 1M` ✓ (`-78.8%`)** |

---

## 3. 🛠️ Production Readiness Checklist

- [ ] **Enable `IBGDA` (`NVSHMEM_IB_ENABLE_IBGDA=1`)** for MoE Decode pools so GPU SMs trigger RDMA writes directly.
- [ ] **Enable Two-Batch Ping-Pong Overlap (`TBO`)** to hide `DeepEP` Dispatch/Combine behind `MLA` & Expert GEMMs.
- [ ] **Reserve Redundant Expert Slots via `EPLB`** (`e.g., 288–320` physical expert slots for `256` logical experts).
