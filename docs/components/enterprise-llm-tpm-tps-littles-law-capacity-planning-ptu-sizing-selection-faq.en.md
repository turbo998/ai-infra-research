# Enterprise LLM `TPM (Tokens Per Minute)` vs. `TPS (Tokens Per Second)` Conversion, `Little's Law` Concurrency Sizing & Cloud `PTU / GSU` Capacity Planning Guide

> **Language**: [中文](./enterprise-llm-tpm-tps-littles-law-capacity-planning-ptu-sizing-selection-faq.md) | **English (Current)**  
> **Audience**: Cloud & AI Infrastructure Solution Architects, Capacity Planners, and FinOps Leads  
> **Provenance**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)

![Enterprise LLM TPM vs TPS & Little's Law Capacity Planning Summary](../../assets/svg/enterprise-llm-tpm-tps-littles-law-capacity-planning-ptu-sizing-selection-faq-summary.svg)

---

## 1. 📌 Executive Summary

> **Bottom Line**: **Enterprise LLM capacity planning requires strictly separating three throughput tiers: (1) Per-User Streaming TPS ($\text{TPS}_{\text{user}} = 1 / \text{TPOT}$), (2) System/Replica Throughput ($\text{TPS}_{\text{sys, out}} = B_{\text{active}} / \text{TPOT} = \text{RPS} \times \bar{N}_{\text{out}}$), and (3) Minute Quota ($\text{TPM} = \text{TPS} \times 60$, where 1 Output Token consumes `4–8x` the hardware budget of 1 Input Token in cloud `PTU / GSU` and GPU sizing: $\text{TPM}_{\text{weighted}} = \text{TPM}_{\text{in}} + w_{\text{out}} \times \text{TPM}_{\text{out}}$). Furthermore, cluster KV memory and queueing are governed by Little's Law ($B_{\text{active}} = \text{RPS}_{\text{peak}} \times \text{E2EL}$). Pairing Little's Law peak concurrency modeling with a `P80 Reserved Base (PTU / Dedicated GPUs) + P20 Serverless PAYGO Spillover` architecture eliminates peak-hour `HTTP 429` throttling (`0.0%`), raises base utilization to `86.4%`, and cuts monthly TCO by `65.2%`!**

- **Interactive `TPM / TPS / Little's Law` Sizer**: [Open Interactive Sizer (`enterprise-llm-tpm-tps-littles-law-capacity-planning-ptu-sizing-sizer.html`)](../reports/enterprise-llm-tpm-tps-littles-law-capacity-planning-ptu-sizing-sizer.html)
- **7-Card 4:5 Visual Report**: [Open 7-Card Report (`enterprise-llm-tpm-tps-littles-law-capacity-planning-ptu-sizing-selection-cards.html`)](../reports/enterprise-llm-tpm-tps-littles-law-capacity-planning-ptu-sizing-selection-cards.html)

---

## 2. 🎯 Architect Deep-Dive Q&A

### Q1: How do `Per-User TPS`, `System TPS`, `Raw TPM`, and `Cloud Weighted PTU TPM` convert into one another?
1. **Per-User Output TPS**: $\text{TPS}_{\text{user}} = \frac{1000}{\text{TPOT (ms)}}$ (`16ms` TPOT = `62.5 tok/s/user`).
2. **System Input & Output TPS**: $\text{TPS}_{\text{sys, in}} = \text{RPS} \times \bar{N}_{\text{in}}$, $\text{TPS}_{\text{sys, out}} = \text{RPS} \times \bar{N}_{\text{out}}$.
3. **Raw TPM**: $\text{TPM}_{\text{in}} = \text{TPS}_{\text{sys, in}} \times 60$, $\text{TPM}_{\text{out}} = \text{TPS}_{\text{sys, out}} \times 60$.
4. **Cloud Provisioned Weighted TPM (`PTU / GSU`)**:
   $$\text{TPM}_{\text{weighted}} = \text{TPM}_{\text{in}} + w_{\text{out}} \times \text{TPM}_{\text{out}} \quad (w_{\text{out}} \approx 4 \text{ to } 8)$$

### Q2: How does Little's Law ($B_{\text{active}} = \text{RPS}_{\text{peak}} \times \text{E2EL}$) govern active KV Cache slots?
$$\text{E2EL} = \text{TTFT} + (\bar{N}_{\text{out}} - 1) \times \text{TPOT}, \qquad B_{\text{active}} = \text{RPS}_{\text{peak}} \times \text{E2EL}$$
At $\text{RPS}_{\text{peak}} = 50\text{ req/s}$, $\text{TTFT} = 0.8\text{s}$, $\bar{N}_{\text{out}} = 500$, and $\text{TPOT} = 16\text{ms}$: $\text{E2EL} = 8.8\text{s}$ and $B_{\text{active}} = 50 \times 8.8 = \mathbf{440}$ concurrent active sessions in GPU KV memory!

---

## 3. 📊 4-Tier Enterprise Capacity Planning Matrix (50K DAU / 50 Peak RPS)

| Metric | Tier 1: Daily-Avg RPS & Unweighted TPM | Tier 2: 100% Serverless PAYGO API | Tier 3: 100% Peak-Sized Static PTU/GPU | Tier 4 (Recommended): Little's Law + `P80 Base / P20 PAYGO Spillover` |
| :--- | :--- | :--- | :--- | :--- |
| **Peak-Hour HTTP 429 Rate** | `58.4%` ✕ | `12.6%` ✕ | `0.0%` ✓ | **`0.0%` ✓** |
| **P99 End-to-End Latency (`E2EL`)** | `34.2 s` ✕ | `14.8 s` | `9.4 s` ✓ | **`8.8 s` ✓** |
| **Base Capacity Utilization** | `31.0%` | `48.0%` | `38.5%` ✕ | **`86.4%` ✓** |
| **Monthly 50-RPS Capacity TCO** | `$385,000 / mo` | `$298,000 / mo` | `$242,000 / mo` | **`$134,000 / mo` (`-65.2%`)** |

---

## 4. 🛠️ Production Readiness Checklist

- [ ] **Multiply Average RPS by Peak-to-Average Ratio ($k_{\text{peak}} = 3.5\text{–}6.0$)** before sizing clusters.
- [ ] **Size KV Concurrency Pool via Little's Law ($B_{\text{active}} = \text{RPS}_{\text{peak}} \times \text{E2EL}$)**.
- [ ] **Apply Output Token Weighting ($w_{\text{out}} = 4\text{–}8$)** when converting Output TPM into cloud PTUs or GPU replicas.
- [ ] **Deploy P80 Reserved Base + P20 PAYGO Spillover** at the AI Gateway layer.
