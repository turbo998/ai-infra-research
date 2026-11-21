# LLM Serving Overload Protection, Predictive Admission Control & Graceful Degradation Guide: Unbounded Queue Collapse vs. Predictive `TTFT` + KV-Watermark Shedding & Automatic SLM Fallback

> **Language**: [中文](./llm-serving-admission-control-backpressure-queue-shedding-ttft-sla-selection-faq.md) | **English (Current)**  
> **Audience**: Cloud & AI Infrastructure Solution Architects and AI Platform SREs  
> **Provenance**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)

![LLM Serving Admission Control Backpressure Queue Shedding Summary](../../assets/svg/llm-serving-admission-control-backpressure-queue-shedding-ttft-sla-selection-faq-summary.svg)

---

## 1. 📌 Executive Summary

> **Bottom Line**: **When a `160 RPS` primary `70B` inference cluster is hit by a `2.2x` flash traffic surge (`320 RPS`, e.g., Monday morning peak) while autoscaled GPU nodes are still booting, an unbounded FIFO queue paired with client timeout retries (`Tier 1`) causes a catastrophic collapse: queued requests wait `18.4s`, clients time out at `8s` and retry, and GPUs waste `68.5%` of their FLOPs computing prefills for already-abandoned "zombie" requests—crashing SLA-compliant `Goodput` (`TTFT <= 800ms`) from `160 RPS` down to `14.0 RPS` (`8.8%` efficiency)! Production AI gateways enforce Predictive `TTFT` & KV-Watermark (`<85%`) Admission Control + End-to-End `X-Request-Deadline` Zombie Cancellation + Graceful Overflow Downgrade to a Fast `8B/32B-FP8` Fallback Pool (`Tier 4`): keeping the primary `70B` pool healthy at `590 ms` P99 `TTFT` while serving overflow traffic seamlessly—delivering `308.0 RPS` total `Goodput` (`22x` higher than unbounded queueing) and a `96.3%` user success rate.**

- **Interactive Overload Protection & Admission Control Simulator**: [Open Interactive Sizer (`llm-serving-admission-control-backpressure-queue-shedding-ttft-sla-sizer.html`)](../reports/llm-serving-admission-control-backpressure-queue-shedding-ttft-sla-sizer.html)
- **7-Card 4:5 Visual Report**: [Open 7-Card Report (`llm-serving-admission-control-backpressure-queue-shedding-ttft-sla-selection-cards.html`)](../reports/llm-serving-admission-control-backpressure-queue-shedding-ttft-sla-selection-cards.html)

---

## 2. 📊 4-Tier Overload Protection & Admission Control Matrix (`320 RPS` Surge on `160 RPS` Primary Cluster)

| Metric | Tier 1: Unbounded FIFO Queue + Client Retry Storm | Tier 2: Static Fixed-RPS Token Bucket | Tier 3: Predictive `TTFT` + `85%` KV Hard `429` Shed | Tier 4 (Recommended): Predictive `TTFT` Gate + `Deadline` Drop + `8B/32B` SLM Fallback |
| :--- | :--- | :--- | :--- | :--- |
| **SLA-Compliant `Goodput`** | `14.0 RPS` ✕ (Collapse!) | `98.0 RPS` | `152.0 RPS` ✓ | **`308.0 RPS` ✓ (`22x` vs Tier 1)** |
| **Primary Pool Capacity Kept** | `8.8%` ✕ | `61.3%` | `95.0%` ✓ | **`96.3%` ✓** |
| **Admitted Request P99 `TTFT`** | `18,400.0 ms` ✕ | `2,450.0 ms` ✕ | `680.0 ms` ✓ | **`590.0 ms` ✓ (`<=800ms` SLA)** |
| **Zombie Request GPU Waste** | `68.5%` ✕ | `18.0%` | `0.5%` ✓ | **`0.0%` ✓ (`Deadline` enforced)** |
| **User Success Rate in Surge** | `4.4%` ✕ | `30.6%` | `47.5%` (`52.5%` get `429`) | **`96.3%` ✓ (Graceful downgrade)** |

---

## 3. 🛠️ Production Readiness Checklist

- [ ] **Gate Admission on Queued Prefill Tokens ($\hat{T}_{\text{wait}} = \sum L_{\text{queued}} / \text{TPS}_{\text{prefill}} < 0.75 \times \text{SLA}_{\text{TTFT}}$) and KV Cache Watermark (`<85%`)** rather than request count alone.
- [ ] **Propagate `X-Request-Deadline` Headers to Drop Expired Queue Entries Before Prefill**, and configure automatic overflow routing (`70B -> 8B/32B-FP8` + RAG Top-K trimming) during traffic surges.
