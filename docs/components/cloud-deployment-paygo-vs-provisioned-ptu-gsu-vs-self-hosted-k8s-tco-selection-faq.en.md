# Enterprise Cloud LLM Deployment Mode & TCO Selection Guide: `Serverless API (MaaS PAYGO)` vs. `Provisioned Throughput (Azure PTU / Bedrock / Vertex GSU)` vs. `Self-Hosted K8s (AKS/GKE + KAITO/vLLM)`

> **Language**: [中文](./cloud-deployment-paygo-vs-provisioned-ptu-gsu-vs-self-hosted-k8s-tco-selection-faq.md) | **English (Current)**  
> **Audience**: Cloud & AI Infrastructure Solution Architects, CTOs, and AI FinOps Leads  
> **Provenance**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)

![Cloud Deployment TCO PAYGO vs PTU vs Self-Hosted K8s Summary](../../assets/svg/cloud-deployment-paygo-vs-provisioned-ptu-gsu-vs-self-hosted-k8s-tco-selection-faq-summary.svg)

---

## 1. 📌 Executive Summary

> **Bottom Line**: **Anchor enterprise deployment decisions on two quantitative break-even thresholds: (1) `Provisioned Throughput (PTU / GSU)` beats `Serverless PAYGO` once 24×7 average utilization exceeds **`35%`** (`22%` when leveraging Prefix Cache discounts) while eliminating multi-tenant noisy-neighbor latency spikes; and (2) `Self-Hosted Kubernetes (AKS/GKE + KAITO + vLLM/SGLang)` beats managed APIs once monthly volume exceeds **`2.5 Billion Tokens/Month`** (`>=8` H100/H200 GPUs), where per-token hardware savings amortize the `~$22,000/month` fixed SRE engineering and multi-AZ HA floor. At `15B` tokens/month, a 3-Zone Hybrid FinOps architecture (`70%` self-hosted open-weights on K8s + `20%` P80 PTU + `10%` PAYGO burst spillover) cuts monthly TCO by `71.7%` (`$360K -> $102K/mo`, `$0.68/1M` tokens)!**

- **Interactive Deployment TCO Crossover Calculator**: [Open Interactive Sizer (`cloud-deployment-paygo-vs-provisioned-ptu-gsu-vs-self-hosted-k8s-tco-sizer.html`)](../reports/cloud-deployment-paygo-vs-provisioned-ptu-gsu-vs-self-hosted-k8s-tco-sizer.html)
- **7-Card 4:5 Visual Report**: [Open 7-Card Report (`cloud-deployment-paygo-vs-provisioned-ptu-gsu-vs-self-hosted-k8s-tco-selection-cards.html`)](../reports/cloud-deployment-paygo-vs-provisioned-ptu-gsu-vs-self-hosted-k8s-tco-selection-cards.html)

---

## 2. 📊 4-Tier Cloud Deployment Comparison Matrix (`15B Tokens / Month`)

| Metric | Tier 1: 100% `Serverless PAYGO` | Tier 2: Premature Small Self-Hosted K8s (`<500M tok/mo`) | Tier 3: 100%-Peak Managed `PTU / GSU` | Tier 4 (Recommended): 3-Zone Hybrid (`70% Self-Host K8s + 20% PTU + 10% PAYGO`) |
| :--- | :--- | :--- | :--- | :--- |
| **Effective Cost per 1M Tokens** | `$2.40 / 1M` | `$4.85 / 1M` ✕ (SRE tax!) | `$1.38 / 1M` ✓ | **`$0.68 / 1M` ✓ (`-71.7%`)** |
| **P99 `TTFT` Latency** | `1,480 ms` ✕ | `580 ms` | `490 ms` ✓ | **`420 ms` ✓** |
| **Platform Availability SLA** | `98.4%` (429 throttling) | `99.1%` | `99.9%` ✓ | **`99.95%` ✓** |
| **Monthly `15B`-Token TCO** | `$360,000 / mo` | `$265,000 / mo` | `$207,000 / mo` | **`$102,000 / mo` ✓** |

---

## 3. 🛠️ Production Readiness Checklist

- [ ] **Stay on `Serverless PAYGO` Below `1B–2B` Tokens/Month** unless strict air-gapped data residency mandates in-VPC deployment.
- [ ] **Convert Base Proprietary Traffic to `P80 PTU/GSU + P20 PAYGO Spillover`** once average utilization crosses `35%`.
- [ ] **Adopt `AKS KAITO` or `GKE AI Hypercomputer` for Self-Hosted Open-Weights** above `2.5B` tokens/month.
