# Enterprise AI FinOps Showback/Chargeback, Shared `PTU` Amortization & Progressive Budget Downgrade (`80% -> 95% -> 100%`) Architecture Selection Guide: How to Fairly Split a Shared `$180K/Month` `PTU + GPU` Bill Across 45 Business Units Without Mid-Month `HTTP 429` Outages?

> **Language / 语言**: **English (Current)** | [中文](./enterprise-ai-finops-showback-chargeback-token-budget-governance-selection-faq.md)  
> **Scope**: End-to-end production reference architecture and FinOps governance guide for Cloud & AI Solution Architects, Enterprise FinOps Leads, and AI Platform Engineering Directors designing multi-tenant token chargeback (`FinOps FOCUS 1.0`), time-of-day `PTU` pricing, and graceful model downgrade policies.  
> **Provenance**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`). Deterministic evaluation model sealed in [`experiments/enterprise_ai_finops_showback_chargeback_token_budget_governance_selection/results.json`](../../experiments/enterprise_ai_finops_showback_chargeback_token_budget_governance_selection/results.json).

![Enterprise AI FinOps Chargeback, PTU Amortization & Budget Governance Summary](../../assets/svg/enterprise-ai-finops-showback-chargeback-token-budget-governance-selection-faq-summary.svg)

---

## ① 📌 Executive Summary (3-Minute Architect Verdict)

Once an enterprise consolidates 45 business units onto a shared **AI Gateway & Internal MaaS Platform (`100B tokens/month` across `Azure OpenAI PTU`, `AWS Bedrock`, and `Kubernetes GPU` clusters)**, the biggest operational bottleneck shifts from infrastructure to **FinOps cost governance and quota fairness**. CFOs and platform teams routinely hit three painful failure modes:
1. **Trap #1: The Unmetered Central IT "Free Buffet" Anti-Pattern (Tier 1, `0.0%` Cost Attribution, `64.0%` Monthly Budget Overrun, `$295,000/month`)**: When Central IT hands out a shared API key with zero chargeback, every engineering team defaults to the most expensive frontier reasoning model (`GPT-4o / Claude 3.7 / o3`) even for trivial log parsing—creating a classic tragedy of the commons;
2. **Trap #2: Siloed Per-Department `PTU`s + Hard `100%` Budget Cutoff (`HTTP 429` Cliff, Tier 2, `39.0%` Pool Utilization, `14.2%` Mid-Month Business Outage)**: Enforcing a hard `HTTP 429` block the moment a business unit hits 100% of its monthly quota on the 22nd of the month shuts down customer-facing Copilots for the last 8 days of every month;
3. **Trap #3: Flat Unweighted Token-Count Showback (Tier 3, `58.0%` Attribution Accuracy)**: Charging the exact same per-token rate for a 2 AM cached batch prompt (`90%` prefix cache hit on idle night-time PTUs) as a 10 AM peak uncached reasoning output (`4x–8x` higher GPU memory-bandwidth cost) penalizes well-architected teams and discourages off-peak PTU backfill!

Evaluated across **100B monthly enterprise tokens shared by 45 business units**, the four AI FinOps governance architectures compare as follows:

| Architectural Tier | Chargeback Model & Quota Enforcement | Attribution Accuracy | Budget Overrun Rate | Quota-Cliff Outage Rate | Shared `PTU/GPU` Util | 100B-Token Monthly Bill | Production Verdict |
|---|---|---|---|---|---|---|---|
| **🟥 Tier 1: Unmetered Central IT Key (`Free Buffet`)** | Shared subscription with zero chargeback or quotas | `0.0%` ✕ *(blind spend)* | `64.0%` ✕ *(frontier abuse)* | `18.5%` ✕ *(noisy neighbor)* | `48.0%` | `$295,000 / mo` ✕ | ⛔ Severe budget overruns & noisy-neighbor throttling |
| **🟨 Tier 2: Siloed `PTU`s + Hard `100%` `HTTP 429` Cutoff** | Isolated per-BU PTUs; hard-blocks traffic at 100% cap | `92.0%` | `0.0%` | `14.2%` ✕ *(mid-month outage)* | `39.0%` ✕ *(silo waste)* | `$268,000 / mo` ✕ | ⛔ Hard 429 cutoffs halt live production apps mid-month |
| **🟦 Tier 3: Flat Token-Count Showback (`In + Out`)** | Splits bill by raw token count without cache/peak weights | `58.0%` ✕ *(unfair split)* | `28.0%` | `6.5%` | `62.0%` | `$198,000 / mo` | ⚠️ Penalizes cached & night-batch workloads |
| **🟩 Tier 4: `FOCUS 1.0` Weighted Chargeback + `0.25x` Night `PTU` + `95%` Graceful Downgrade + Loop Breaker** | **`0.15x Cache / 1x In / 4x Out` × `Peak/Night (1.2x vs 0.25x)` + `95%` Auto-Downgrade to `8B SLM` + `$2.50/Trace` Breaker** | **`99.6%` 🏆** *(CFO-grade)* | **`0.8%` 🏆** *(predictable)* | **`0.0%` 🏆** *(zero hard cliff)* | **`89.5%` 🏆** *(night backfill)* | **`$118,000 / mo` 🏆** *(-60.0% vs Tier 1)* | 🌟 **Recommended Enterprise AI FinOps Standard** |

> 💡 **Golden Rule for Enterprise AI FinOps Architects**: **"Never split a shared PTU/GPU bill by raw `Input + Output` token count, and never hard-block production apps with `HTTP 429` at 100% budget! Meter consumption in `Standard Compute Credits` (`0.15x Cached Input + 1.0x Uncached Input + 4.0x Output` × `1.2x Daytime Peak vs. 0.25x Night PTU Backfill`), enforce a `$2.50` per-trace circuit breaker to kill runaway agent loops, and automatically downgrade non-P0 traffic to an `8B / Mini SLM` (`-94%` marginal cost) when a department reaches `95%` of its monthly budget."**

---

## ② 🎯 Customer & Architect Deep-Dive Q&A (3 Shared PTU Chargeback, Mid-Month Quota Cliff & Runaway Agent Loop Pain Points)

### Q1: Since monthly `Azure OpenAI PTU` and `K8s GPU` reservations have a fixed monthly cost, what exact mathematical formula should we use to charge back 45 departments fairly?
- **The `FinOps FOCUS 1.0` Compute-Weighted & Time-of-Day Credit Formula**:
  $$\text{Credits}_i = \left(0.15 \times T_{\text{cached}} + 1.0 \times T_{\text{input}} + 4.0 \times T_{\text{output}}\right) \times M_{\text{model}} \times W_{\text{time}}$$
  - **Why `0.15x : 1.0x : 4.0x`?**: Autoregressive decode (`Output`) is memory-bandwidth-bound and consumes ~4x more GPU time per token than uncached prefill (`Input`), while prefix-cached tokens require near-zero compute (`0.15x`).
  - **Why `1.2x Daytime Peak` vs. `0.25x Night Backfill` (`20:00–07:00`)?**: Offering a **75% credit discount for night-time execution** incentivizes engineering teams to schedule bulk document extraction and eval runs overnight—driving 24x7 shared PTU utilization from `48.0%` up to **`89.5%`**!

### Q2: How do we prevent a business unit from overspending its monthly budget without causing a `14.2%` mid-month service outage (`HTTP 429` cliff)?
- **3-Stage Progressive Budget Governance (`80% Alert -> 95% Graceful Model Downgrade -> 100% P0-Only`)**:
  1. **At `80%` Monthly Budget**: Dispatch automated FinOps alerts to the BU owner and tighten semantic cache hit thresholds;
  2. **At `95%` Monthly Budget (Graceful Model Downgrade)**: Instead of returning `HTTP 429`, the AI Gateway transparently routes non-P0 requests from frontier models (`GPT-4o / Claude 3.7` at `$2.50–$10/1M`) down to **`GPT-4o-mini / Qwen2.5-8B-Instruct` (`$0.15/1M` — `94%` cheaper!)** with header `X-FinOps-Downgraded: true`. The remaining 5% budget easily covers the last week of the month with **`0.0%` downtime**!
  3. **At `100%` Monthly Budget**: Pause `dev/staging` and async batch queues while keeping `P0` customer-facing traffic live on the SLM tier.

### Q3: How do we stop a single buggy multi-agent retry loop from burning `$12,000` over a weekend before anyone notices on Monday morning?
- **Per-Trace Micro-Caps (`$2.50 / trace_id`) + 3-Repeat Tool-Call Hash Circuit Breakers**:
  - Coarse monthly department alerts fire far too late to stop an infinite recursion bug.
  - The AI Gateway tracks real-time spend per `trace_id / session_id` in Redis: if any single session exceeds **`25 turns`**, **`$2.50` cumulative spend**, or **`>= 3` consecutive identical tool-call argument hashes**, the gateway halts the trace in `< 15 seconds` (capping the weekend loss at **`$0.42`** instead of `$12,000`!).

---

## ③ 🧭 Visual Architecture Diagrams (Decision Tree + `FOCUS 1.0` Chargeback & Downgrade Topology + `95%` Budget Sequence)

### 3.1 Enterprise AI FinOps Chargeback & Budget Governance Selection Decision Tree

```mermaid
flowchart TD
    A["Enterprise AI FinOps & Quota Governance Selection<br/>(100B Tokens/Month Across 45 Business Units)"] --> B{"How are shared PTU, PAYGO, and GPU cluster costs governed?"}
    B -- "Unmetered Central IT shared API key (Tier 1)" --> C["⛔ $295K/Mo 'Free Buffet' Tragedy!<br/>64% budget overrun as every team defaults to frontier models"]
    B -- "Siloed per-BU PTUs + Hard 100% HTTP 429 cutoff (Tier 2)" --> D["⛔ 39% Utilization & 14.2% Mid-Month Outages!<br/>Customer apps crash on the 22nd of the month when quota hits 100%"]
    B -- "Shared Multi-Tenant Pool with Departmental Chargeback" --> E{"What accounting formula & budget enforcement policy is used?"}
    E -- "Flat unweighted Input+Output token count (Tier 3)" --> F["⚠️ 58% Attribution Accuracy!<br/>Penalizes cached prompts & night batch jobs; subsidizes peak hogs"]
    E -- "FOCUS 1.0 Weighted Credits + 0.25x Night PTU + 95% Auto-Downgrade (Tier 4)" --> G["🟩 Tier 4: 0.15x/1x/4x × Peak/Night Credits + 95% Graceful SLM Downgrade 🏆<br/>(99.6% Fairness | 0.0% Outage | 89.5% PTU Util | -60.0% Monthly Bill)"]
    style G fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#f8fafc
    style C fill:#7f1d1d,stroke:#ef4444,color:#fecaca
    style D fill:#7f1d1d,stroke:#ef4444,color:#fecaca
```

### 3.2 Tier 4 `FOCUS 1.0` Compute-Weighted Chargeback & Graceful Model Downgrade Reference Architecture

```mermaid
flowchart LR
    subgraph Ingress["1. Tagged Multi-Tenant Request Ingress"]
        Req["📩 Mandatory Headers:<br/>x-dept-id / x-app-id / x-criticality<br/>+ OTel trace_id"]
    end

    subgraph Policy["2. AI Gateway Real-Time Budget & Loop Guard"]
        Req --> Loop["🛡️ Per-Trace Loop Circuit Breaker (Redis)<br/>Halts trace if spend > $2.50 or 3x identical tool call"]
        Loop --> Gate["🎚️ 80% / 95% / 100% Progressive Budget Router<br/><95%: Frontier Pool (GPT-4o / Claude 3.7)<br/>>=95%: Graceful Downgrade to 8B/Mini SLM (-94% cost)"]
    end

    subgraph FinOps["3. OpenTelemetry GenAI -> FOCUS 1.0 Billing Pipeline"]
        Gate --> Compute["☁️ Shared PTU + K8s vLLM Pool<br/>(89.5% 24x7 Utilization)"]
        Compute --> Ledger["📊 FOCUS 1.0 Weighted Chargeback Ledger<br/>0.15x Cache / 1x In / 4x Out<br/>× (1.2x Peak | 0.25x Night PTU Backfill)"]
    end
```

### 3.3 End-to-End Sequence: Seamless `95%` Budget Graceful Downgrade Without `HTTP 429` Business Outage

```mermaid
sequenceDiagram
    autonumber
    participant App as Marketing Copilot (dept=mkt, P2)
    participant GW as AI Gateway (FinOps Governance Filter)
    participant Redis as Real-Time Budget & Trace State
    participant Frontier as Frontier Pool (GPT-4o / Claude, $5/1M)
    participant SLM as Cost-Efficient SLM Pool (Qwen2.5-8B / 4o-mini, $0.15/1M)

    App->>GW: 1. Send request on Nov 22 (requested model: gpt-4o, trace_id: tr-8821)
    GW->>Redis: 2. Check tr-8821 spend ($0.08 < $2.50 cap) & dept=mkt monthly budget consumption
    Redis-->>GW: 3. Return dept=mkt at 95.4% of monthly budget (>= 95% downgrade threshold!)
    Note over GW: Trigger Tier-4 Graceful Model Downgrade instead of HTTP 429!<br/>Transparently route P2 request to 8B/Mini SLM (-94% marginal token cost)
    GW->>SLM: 4. Execute inference on 8B/Mini SLM pool
    SLM-->>GW: 5. Return response in 310ms (attach header `X-FinOps-Downgraded: true`)
    GW->>Redis: 6. Deduct weighted compute credits (0.15x/1x/4x × ToD factor)
    GW-->>App: 7. Zero mid-month business outage (0.0%) and <0.8% monthly budget variance!
```

---

## ④ 📊 4-Tier Architecture Decision Matrix

| Evaluation Dimension | 🟥 Tier 1: Unmetered `Free Buffet` | 🟨 Tier 2: Siloed `PTU` + Hard `429` | 🟦 Tier 3: Flat Token Showback | 🟩 Tier 4: `FOCUS 1.0` + `95%` Graceful Downgrade |
|---|---|---|---|---|
| **Cost Attribution Accuracy** | `0.0%` ✕ (no attribution) | `92.0%` | `58.0%` ✕ (ignores cache/ToD) | 🏆 **`99.6%` (`0.15x/1x/4x` × Peak/Night weights)** |
| **Monthly Budget Overrun Rate** | `64.0%` ✕ (uncontrolled spend) | `0.0%` | `28.0%` | 🏆 **`0.8%` (`95%` downgrade + `$2.50` trace cap)** |
| **Quota-Cliff Business Outage** | `18.5%` ✕ (noisy-neighbor 429s) | `14.2%` ✕ (hard 100% cutoff) | `6.5%` | 🏆 **`0.0%` (graceful SLM fallback keeps apps live)** |
| **Shared `PTU/GPU` Utilization** | `48.0%` | `39.0%` ✕ (stranded silos) | `62.0%` | 🏆 **`89.5%` (`0.25x` night discount drives backfill)** |
| **100B-Token Monthly AI Bill** | `$295,000 / mo` ✕ | `$268,000 / mo` ✕ | `$198,000 / mo` | **`$118,000 / mo` 🏆 (-60.0% vs Tier 1)** |

---

## ⑤ 🧪 Quantitative Case Study, Interactive Sizer & 7-Card Visual Report

- **Interactive HTML Sizer**: 👉 [Open Enterprise AI FinOps Chargeback, PTU Amortization & Budget Governance Sizer](../reports/enterprise-ai-finops-showback-chargeback-token-budget-governance-sizer.html)
- **7-Card 4:5 Visual Benchmark Report**: 👉 [Open 7-Card Visual Benchmark Deck](../reports/enterprise-ai-finops-showback-chargeback-token-budget-governance-selection-cards.html)
- **Deterministic Verification**:
  ```bash
  python experiments/enterprise_ai_finops_showback_chargeback_token_budget_governance_selection/run.py --check
  python -m unittest tests/test_enterprise_ai_finops_showback_chargeback_token_budget_governance_selection.py
  ```

---

## ⑥ 🛠️ Production Readiness & Architecture Review Checklist

- [ ] **Mandate `[x-dept-id, x-app-id, x-env, x-criticality]` Headers on Every AI Gateway Request**: Reject untagged API calls at ingress and emit `OpenTelemetry GenAI` usage spans into a `FinOps FOCUS 1.0` billing table.
- [ ] **Meter Compute Credits Using `0.15x Cached Input : 1.0x Uncached Input : 4.0x Output` Weights**: Accurately reflect GPU prefill vs. decode hardware costs and reward teams that implement prompt caching.
- [ ] **Offer a `0.25x` Off-Peak Night Discount (`20:00–07:00`) on Shared `PTU / GPU` Pools**: Incentivize engineering teams to shift batch indexing and evaluation workloads into idle night hours (`89.5%` 24x7 utilization).
- [ ] **Replace Hard `100% HTTP 429` Cutoffs with an `80% Alert -> 95% Graceful SLM Downgrade -> 100% P0-Only` Policy**: Cut marginal token cost by `94%` during the final 5% of monthly budget without taking down production apps.
- [ ] **Enforce Per-Trace Circuit Breakers (`max_cost <= $2.50`, `max_turns <= 25`, `< 3` identical tool-call hashes)**: Halt runaway recursive agent loops in under 15 seconds.
