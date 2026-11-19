# Enterprise Cloud-Managed LLM Deployment Modes (`Serverless PAYGO` vs. `Provisioned Throughput PTU`) vs. Self-Hosted Kubernetes (`AKS/EKS + KAITO/vLLM`) Cost-Inflection, SLA & Team-Readiness Selection Guide

> **Language / 语言**: **English (Current)** | [中文](./enterprise-cloud-managed-paas-ptu-paygo-vs-self-hosted-k8s-tco-selection-faq.md)  
> **Scope**: End-to-end production reference architecture and full-TCO break-even guide (including Platform SRE FTE payroll) for Cloud & AI Solution Architects, CTOs, VPs of Infrastructure, and FinOps Leads.  
> **Provenance**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`). Deterministic evaluation model sealed in [`experiments/enterprise_cloud_managed_paas_ptu_paygo_vs_self_hosted_k8s_tco_selection/results.json`](../../experiments/enterprise_cloud_managed_paas_ptu_paygo_vs_self_hosted_k8s_tco_selection/results.json).

![Cloud Managed (PAYGO / PTU) vs. Self-Hosted K8s (AKS + KAITO/vLLM) TCO Summary](../../assets/svg/enterprise-cloud-managed-paas-ptu-paygo-vs-self-hosted-k8s-tco-selection-faq-summary.svg)

---

## ① 📌 Executive Summary (3-Minute Architect Verdict)

The single most common strategic question CTOs and CFOs ask Cloud & AI Solution Architects is: **"When should we use per-token `Serverless PAYGO` APIs, when should we commit to `Provisioned Throughput Units (PTU)` (`Azure OpenAI PTU / AWS Bedrock Provisioned`), and at what exact volume does self-hosting `vLLM / SGLang` on Kubernetes (`AKS / EKS`) actually save money?"** Three widespread TCO calculation mistakes cost enterprises millions:
1. **Mistake #1: Staying 100% on `Serverless PAYGO` at High Sustained Scale (`120B Tokens/Month`, Tier 1, `$322,000/month`, `14.5%` Peak HTTP 429 Throttling, `2,380ms` P99 TTFT)**: Multi-tenant PAYGO endpoints have the highest unit cost at scale (`$2.65/1M tokens`) and suffer noisy-neighbor latency spikes during peak business hours;
2. **Mistake #2: Sizing Fixed `PTU` Reservations for 100% Monday 10:00 AM Peak Load (Tier 2, `68.0%` Overnight/Weekend Idle Waste, `$244,000/month`)**: Buying 220 PTUs to cover a 2-hour daytime peak when average 24x7 load is 95 PTUs;
3. **Mistake #3: Premature Raw DIY Kubernetes Self-Hosting Without Counting the `$55,500/Month` GPU SRE Payroll Tax (Tier 3)**: Comparing raw hourly H100 rental against API token prices while ignoring the **2.5–3.0 full-time Platform SREs (`~$55,500/month` fully loaded payroll)** needed to maintain CUDA drivers, NCCL networking, vLLM upgrades, and 24x7 on-call!

Evaluated across a **120-Billion-token/month enterprise portfolio (`40%` frontier proprietary reasoning + `60%` high-volume open-weight/distilled tasks, including `$18,500/mo` per Platform SRE FTE)**, the four deployment architectures compare as follows:

| Architectural Tier | Core Deployment Mode Mix | Blended Cost / 1M Tokens | P99 `TTFT` SLA | Peak `HTTP 429` Rate | Platform SRE Staffing | Full Monthly TCO (Incl. SRE) | Production Verdict |
|---|---|---|---|---|---|---|---|
| **🟥 Tier 1: 100% Serverless `PAYGO` API** | All 120B tokens routed to per-token PAYGO APIs | `$2.65 / 1M` ✕ | `2,380 ms` ✕ *(noisy neighbor)* | `14.5%` ✕ *(peak throttling)* | `0.2 FTE` (`$3.7K/mo`) | `$322,000 / mo` ✕ | ⚠️ Great for `< 5B tok/mo`; too costly/spiky at 120B |
| **🟨 Tier 2: 100% Peak-Sized Cloud `PTU`** | Fixed PTUs sized for 100% peak without burst spillover | `$1.95 / 1M` | `480 ms` 🏆 *(dedicated capacity)* | `0.8%` | `0.5 FTE` (`$9.3K/mo`) | `$244,000 / mo` ✕ *(68% idle waste)* | ⛔ Overpays for 24x7 idle peak headroom |
| **🟦 Tier 3: Raw DIY K8s + `vLLM` (`3.0 SRE FTEs`)** | Bespoke K8s GPU cluster + 3 full-time GPU SREs | `$1.62 / 1M` | `620 ms` | `2.4%` | `3.0 FTEs` ✕ *($55.5K/mo payroll!)* | `$194,000 / mo` | ⚠️ Heavy SRE payroll tax eats up GPU savings |
| **🟩 Tier 4: Volume-Inflection Hybrid (`P55 PTU + PAYGO Burst` + `AKS KAITO vLLM`)** | **40% frontier on `P55 Base PTU + PAYGO Burst` (`88%` util) + 60% open-weight on `AKS + KAITO vLLM` (`1.0 FTE`)** | **`$0.72 / 1M` 🏆** *(-72.8%)* | **`420 ms` 🏆** *(guaranteed SLA)* | **`0.0%` 🏆** *(seamless overflow)* | **`1.0 FTE` 🏆** *(KAITO automation)* | **`$106,000 / mo` 🏆** *(-67.1%)* | 🌟 **Recommended Enterprise Deployment Roadmap** |

> 💡 **Golden Rule for Enterprise LLM Deployment Mode Selection (The Three Volume Inflection Zones)**:
> - **Zone 1 (`< 5 Billion Tokens/Month` or Spiky `< 35%` Duty Cycle) 👉 `100% Serverless PAYGO`**: Zero idle commitment and `0.2 FTE` ops overhead; never buy PTUs or spin up K8s here.
> - **Zone 2 (`5B–30B Tokens/Month` on Frontier Models at `>= 50%` Duty Cycle) 👉 `P55 Base PTU + PAYGO Burst Spillover`**: Provision PTUs at the **55th-percentile (`P55`) diurnal baseline** (keeping PTUs `88%` utilized) and configure your AI Gateway to spill the top 15% daytime peaks over to PAYGO (`0.0%` 429s, `420ms` P99 TTFT).
> - **Zone 3 (`> 25B–30B Tokens/Month` on Open-Weight / Distilled `8B–70B` Models) 👉 `Self-Hosted AKS / EKS + KAITO / vLLM`**: Above 30B tokens/month, GPU compute savings decisively outweigh the `1.0 FTE` (`$18.5K/mo`) staffing cost of a managed Kubernetes AI operator (`AKS KAITO`), dropping blended cost to **`$0.72 / 1M tokens`**!

---

## ② 🎯 Customer & Architect Deep-Dive Q&A (3 Break-Even Math, PTU Sizing & Hidden SRE Cost Pain Points)

### Q1: Why do naive "GPU Rental vs. API Price" spreadsheets mislead CTOs, and why is `~25B–30B Tokens/Month` the true break-even threshold for self-hosted Kubernetes?
- **Including the Hidden `$55,500/Month` Platform SRE Payroll Tax**:
  - Renting raw GPUs without a managed operator requires ~`3.0 FTEs` (`$55,500/month`) for driver upgrades, CVE patching, model P2P caching, and 24x7 on-call. If your workload is only `5B tokens/month`, a Serverless API costs `$13,000/month` total—whereas a DIY K8s cluster costs `$15,000` in GPUs **plus `$55,500` in SRE salaries (`$70,500/month`)**!
  - Using **`Azure AKS KAITO (Kubernetes AI Toolchain Operator)`** automates GPU node provisioning, preset OCI weight streaming, and vLLM configuration, cutting ops staffing from `3.0 FTEs` to **`1.0 FTE ($18,500/mo)`** and making self-hosting highly profitable above `25B–30B tokens/month`.

### Q2: Why should reserved `PTUs` always be sized at the `P55` baseline load paired with `Serverless PAYGO` burst overflow rather than at 100% peak load?
- **The Diurnal Base + Peak Area Law**:
  - Sizing PTUs for the Monday 10:00 AM spike (`220 PTUs`, Tier 2) leaves 68% of reserved capacity idle overnight and on weekends.
  - Sizing at **`P55` (`110 PTUs`, Tier 4)** keeps the reserved pool running at `88%` utilization while the AI Gateway (`Azure APIM / Envoy`) transparently overflows the top 15% daytime peak calls to `Serverless PAYGO` whenever PTU utilization crosses 98%—saving **`$138,000/month`** with **`0.0%` HTTP 429 errors**!

---

## ③ 🧭 Visual Architecture Diagrams (Decision Tree + Hybrid PTU/PAYGO/AKS Topology + Sequence Flow)

### 3.1 Enterprise Deployment Mode Selection Decision Tree (`PAYGO` vs. `PTU` vs. `AKS + KAITO/vLLM`)

```mermaid
flowchart TD
    A["Enterprise LLM Deployment Mode Selection<br/>(Monthly Token Volume + Diurnal Peak Ratio + SLA + SRE Readiness)"] --> B{"Monthly token volume & diurnal duty cycle?"}
    B -- "Zone 1: < 5B Tokens/Mo or Spiky (<35% duty cycle)" --> C["🟦 100% Serverless PAYGO Token API<br/>(Zero fixed commitment | 0.2 SRE FTE | Fast PoC ramp)"]
    B -- "Zone 2 & 3: >= 5B to 120B+ Tokens/Mo (or strict <500ms P99 TTFT SLA)" --> D{"Proprietary Frontier Model or Open-Weight / Distilled Model?"}
    D -- "Proprietary Frontier Model (GPT-4o / Claude / o3-mini)" --> E["🟩 P55 Baseline PTU Reservation + Serverless PAYGO Burst 🏆<br/>(PTUs run at 88% utilization; top 15% peak spills to PAYGO with 0% 429s)"]
    D -- "Open-Weight / Distilled Model (8B–70B, >30B Tokens/Mo)<br/>& >= 1.0 K8s Platform Engineer" --> F["🟩 Self-Hosted AKS / EKS + KAITO Operator (vLLM / SGLang) 🏆<br/>(Cuts ops to 1.0 FTE & drops blended unit cost to $0.72/1M tokens, -67.1% TCO)"]
    style C fill:#1e3a8a,stroke:#3b82f6,stroke-width:2px,color:#f8fafc
    style E fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#f8fafc
    style F fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#f8fafc
```

### 3.2 Tier 4 Volume-Inflection Hybrid Architecture (`P55 PTU + PAYGO Burst` + `AKS KAITO vLLM`)

```mermaid
flowchart LR
    subgraph Gateway["1. Unified AI Gateway & Quota Router"]
        Traffic["120B Tokens/Month<br/>Enterprise Portfolio"] --> AIGW["🌐 AI Gateway (Azure APIM / Envoy / LiteLLM)<br/>Routes by model family, QoS priority & live PTU headroom"]
    end

    subgraph CloudManaged["2. Frontier Proprietary Lane (40% Load = 48B Tok/Mo)"]
        AIGW -->|85% Base Traffic| PTU["💎 P55 Reserved PTU Pool (110 PTUs)<br/>88% 24x7 Utilization | 420ms P99 TTFT SLA"]
        AIGW -.->|15% Peak Burst Overflow (on 429)| PAYGO["☁️ Serverless PAYGO Burst Pool<br/>Absorbs morning peaks (0.0% 429s | Zero idle PTU waste)"]
    end

    subgraph SelfHosted["3. High-Volume Open-Weight Lane (60% Load = 72B Tok/Mo)"]
        AIGW -->|High-Volume Extraction / RAG / Code| AKS["🚀 Self-Hosted AKS / EKS + KAITO (vLLM / SGLang)<br/>H200/L40S Pool + KEDA Autoscaling<br/>(1.0 SRE FTE | $0.45/1M compute cost)"]
    end
```

### 3.3 End-to-End Sequence: Routing Across `AKS KAITO vLLM`, `P55 PTU`, and `PAYGO Burst` During Peak Hours

```mermaid
sequenceDiagram
    autonumber
    participant App as Enterprise Application
    participant GW as AI Gateway (Hybrid Quota Router)
    participant PTU as Cloud P55 Base PTU Pool (110 PTUs)
    participant PAYGO as Cloud Serverless PAYGO Pool
    participant AKS as Self-Hosted AKS + KAITO (vLLM Pool)

    App->>GW: Send inference request with workload & SLA metadata
    alt 60% High-Volume Open-Weight / Distilled Workload (72B tokens/mo > 30B break-even)
        GW->>AKS: Route to self-hosted AKS KAITO `vLLM (8B/70B FP8)` pool
        AKS-->>App: Return response in 410ms at $0.45/1M tokens!
    else 40% Complex Frontier Reasoning Workload (48B tokens/mo)
        alt PTU Pool Utilization <= 98% (85% of daily frontier calls)
            GW->>PTU: Route to reserved 110-PTU baseline pool
            PTU-->>App: Return deterministic 420ms P99 TTFT response!
        else Monday 10:00 AM Peak Exceeds 110 PTUs (top 15% burst)
            GW->>PAYGO: Spill overflow call to Serverless PAYGO in <5ms
            PAYGO-->>App: Serve burst with 0.0% HTTP 429 errors (saves $120K/mo vs buying 220 peak PTUs)!
        end
    end
```

---

## ④ 📊 4-Tier Architecture Decision Matrix

| Evaluation Dimension | 🟥 Tier 1: 100% `PAYGO` | 🟨 Tier 2: 100% Peak `PTU` | 🟦 Tier 3: Raw DIY K8s (`3 FTEs`) | 🟩 Tier 4: `P55 PTU+Burst` + `AKS KAITO vLLM` |
|---|---|---|---|---|
| **Blended Cost / 1M Tokens (incl. SRE)** | `$2.65 / 1M` ✕ | `$1.95 / 1M` | `$1.62 / 1M` | 🏆 **`$0.72 / 1M` (-72.8% vs Tier 1)** |
| **Guaranteed P99 `TTFT` Latency** | `2,380 ms` ✕ (public pool lag) | `480 ms` 🏆 | `620 ms` | 🏆 **`420 ms` (dedicated PTU + local H200/L40S)** |
| **Peak-Hour `HTTP 429` Rate** | `14.5%` ✕ | `0.8%` | `2.4%` | 🏆 **`0.0%` (PTU base + instant PAYGO spillover)** |
| **Platform SRE Staffing (`FTE`)** | `0.2 FTE` (`$3.7K/mo`) | `0.5 FTE` (`$9.3K/mo`) | `3.0 FTEs` (`$55.5K/mo`) ✕ | 🏆 **`1.0 FTE` (`$18.5K/mo` via `AKS KAITO`)** |
| **120B-Token Full Monthly TCO** | `$322,000 / mo` | `$244,000 / mo` | `$194,000 / mo` | **`$106,000 / mo` 🏆 (-67.1%, saves $2.59M/yr)** |

---

## ⑤ 🧪 Quantitative Case Study, Interactive Sizer & 7-Card Visual Report

- **Interactive HTML Sizer**: 👉 [Open Enterprise Cloud-Managed (PAYGO / PTU) vs. Self-Hosted K8s (AKS + vLLM) TCO Sizer](../reports/enterprise-cloud-managed-paas-ptu-paygo-vs-self-hosted-k8s-tco-sizer.html)
- **7-Card 4:5 Visual Benchmark Report**: 👉 [Open 7-Card Visual Benchmark Deck](../reports/enterprise-cloud-managed-paas-ptu-paygo-vs-self-hosted-k8s-tco-selection-cards.html)
- **Deterministic Verification**:
  ```bash
  python experiments/enterprise_cloud_managed_paas_ptu_paygo_vs_self_hosted_k8s_tco_selection/run.py --check
  python -m unittest tests/test_enterprise_cloud_managed_paas_ptu_paygo_vs_self_hosted_k8s_tco_selection.py
  ```

---

## ⑥ 🛠️ Production Readiness & Architecture Review Checklist

- [ ] **Include Fully Loaded Platform SRE Payroll (`~$18.5K/mo per FTE`) in Every Self-Hosted vs. Cloud API TCO Comparison**: Do not self-host Kubernetes GPU clusters for workloads below `25B–30B tokens/month`.
- [ ] **Size Cloud Provisioned Throughput (`PTU`) at the `P55` Diurnal Baseline and Configure Automatic AI Gateway Spillover to `Serverless PAYGO`**: Eliminate the 68% overnight idle waste of peak-sized PTU reservations.
- [ ] **Use Managed Kubernetes AI Operators (`Azure AKS KAITO`) When Self-Hosting Open-Weight Models (`> 30B Tokens/Month`)**: Keep operational staffing at `<= 1.0 FTE` instead of 3.0 DIY SREs.
- [ ] **Abstract All Deployment Modes Behind a Unified AI Gateway (`Azure APIM / Envoy / LiteLLM`)**: Allow applications to transition seamlessly from `PAYGO` (Day 1) to `P55 PTU + Burst` (Day 30) to `AKS + KAITO vLLM` (Day 90) with zero application code changes.
