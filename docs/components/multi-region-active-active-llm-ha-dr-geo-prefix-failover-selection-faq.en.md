# Multi-Region Active-Active LLM HA/DR & Global Traffic Routing Guide: `Geo-Sticky Prefix Affinity`, Preventing Failover KV Cold-Storms, 3-Tier Escape Hatch & `1.8s RTO`

> **Language**: [中文](./multi-region-active-active-llm-ha-dr-geo-prefix-failover-selection-faq.md) | **English (Current)**  
> **Audience**: Cloud & AI Infrastructure Solution Architects and Enterprise SRE / Resilience Leads  
> **Provenance**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)

![Multi-Region Active-Active LLM HA/DR & Geo-Sticky Failover Summary](../../assets/svg/multi-region-active-active-llm-ha-dr-geo-prefix-failover-selection-faq-summary.svg)

---

## 1. 📌 Executive Summary

> **Bottom Line**: **Two classic DR patterns fail catastrophically for enterprise LLMs: (1) "Pilot-Light On-Demand GPU Scale-Up" fails with `OutOfStock` (`88.5%` drop rate) when an entire cloud region goes dark and thousands of customers request `H100/H200` simultaneously; and (2) even a 100%-idle Hot-Standby GPU cluster (`$460K/mo`) suffers a `5.0x` Prefill compute surge (`KV Cache Cold-Storm`, `34.2%` drop rate) when multi-turn `16K` sessions fail over to a region with `0%` cached KV prefixes! Architect multi-region resilience with (1) Active-Active Geo-Sticky Session Routing (`hash(session_id)` pinned to its home region), (2) Asynchronous Top-100 Prefix Warm-Replication + Failover Brownout Context Capping, (3) `40%` Preemptible Batch/SFT Headroom GPUs per region (`4.2s` `vLLM wake_up`), and (4) a Cross-Cloud Serverless `PAYGO` Escape Hatch—achieving `1.8s RTO`, `0.0%` cascading drops, `99.99%` availability, and `-58.3%` monthly TCO (`$192K/mo`)!**

- **Interactive Multi-Region HA/DR Sizer**: [Open Interactive Sizer (`multi-region-active-active-llm-ha-dr-geo-prefix-failover-sizer.html`)](../reports/multi-region-active-active-llm-ha-dr-geo-prefix-failover-sizer.html)
- **7-Card 4:5 Visual Report**: [Open 7-Card Report (`multi-region-active-active-llm-ha-dr-geo-prefix-failover-selection-cards.html`)](../reports/multi-region-active-active-llm-ha-dr-geo-prefix-failover-selection-cards.html)

---

## 2. 📊 4-Tier Multi-Region LLM HA/DR Comparison Matrix (`120 Peak RPS`)

| Metric | Tier 1: Cold Pilot-Light On-Demand GPU DR | Tier 2: Active-Passive 100% Idle Hot Standby | Tier 3: Naive Active-Active `50/50` Round-Robin | Tier 4 (Recommended): `Active-Active Geo-Sticky` + Preemptible Buffer + Escape Hatch |
| :--- | :--- | :--- | :--- | :--- |
| **Regional Outage `RTO`** | `2,400.0 s` (40m ✕) | `75.0 s` | `8.5 s` | **`1.8 s` ✓** |
| **Failover Cascading Drop Rate** | `88.5%` ✕ (`OutOfStock`) | `34.2%` ✕ (KV Cold-Storm) | `19.4%` | **`0.0%` ✓** |
| **Annual Effective Availability** | `99.15%` ✕ | `99.82%` | `99.91%` | **`99.99%` ✓** |
| **Monthly Global HA/DR TCO** | `$395,000 / mo` | `$460,000 / mo` ✕ | `$310,000 / mo` | **`$192,000 / mo` ✓ (`-58.3%`)** |

---

## 3. 🛠️ Production Readiness Checklist

- [ ] **Never Rely on On-Demand GPU Provisioning During a Cloud Regional Outage**: Hold active capacity in both regions and fill spare headroom with preemptible offline batch/SFT jobs.
- [ ] **Enforce Geo-Sticky Session Affinity (`session_id`)** and asynchronously warm top system/RAG prefixes in both regions to prevent `5x` failover KV cold-storms.
