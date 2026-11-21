# Enterprise AI Gateway Token Quota Governance, Multi-BU FinOps Chargeback & Guardrail Selection Guide: `Azure APIM` vs. `Kong AI Gateway` vs. `LiteLLM Enterprise` vs. `Envoy AI Gateway`

> **Language / 语言**: **English (Current)** | [中文](./enterprise-ai-gateway-token-quota-chargeback-apim-kong-litellm-envoy-selection-faq.md)  
> **Scope**: A practical FinOps chargeback, streaming quota governance, and gateway selection guide for Cloud & AI Solution Architects—answering **(1) why legacy HTTP `RPM` rate-limiting and single-phase `max_tokens` pre-deduction fail on LLM traffic, (2) how to implement a Two-Phase Streaming Weighted Token Bucket (`Pre-Flight P80 Soft Reserve` $\to$ `Post-SSE 4-Tuple Exact Reconciliation`), and (3) how Hierarchical Elastic Quota Borrowing (`<80%` pool borrowing + `>=85%` per-BU PAYGO spillover) across `Azure APIM`, `Kong AI`, `LiteLLM Enterprise`, and `Envoy AI Gateway` raises shared pool utilization from `52.5%` to `89.4%` with `0.02%` chargeback error**.  
> **Provenance**: Analytical quota and FinOps model synthesized from published enterprise AI gateway benchmarks (`provenance: synthetic + anonymized_reported`, `independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`). See [`experiments/enterprise_ai_gateway_token_quota_chargeback_apim_kong_litellm_envoy_selection/results.json`](../../experiments/enterprise_ai_gateway_token_quota_chargeback_apim_kong_litellm_envoy_selection/results.json).

![Enterprise AI Gateway Token Quota Governance & FinOps Chargeback Summary](../../assets/svg/enterprise-ai-gateway-token-quota-chargeback-apim-kong-litellm-envoy-selection-faq-summary.svg)

> 🔗 **Interactive H5 Learning Lab & Companion Tools (Zero External Dependencies)**:
> - 📱 **[Interactive H5 Study Page: Two-Phase Weighted Token Bucket, Elastic Quota Borrowing & 4-Gateway FinOps Lab (`docs/h5/ai-gateway-token-quota-chargeback-apim-kong-litellm-lab.html`)](../h5/ai-gateway-token-quota-chargeback-apim-kong-litellm-lab.html)**
> - 🎓 **[AI Infra Panorama H5 Learning Hub (`docs/h5/index.html`)](../h5/index.html)**
> - 🧮 **[Interactive Token Quota & FinOps Chargeback Sizer (`enterprise-ai-gateway-token-quota-chargeback-apim-kong-litellm-envoy-selection-sizer.html`)](../reports/enterprise-ai-gateway-token-quota-chargeback-apim-kong-litellm-envoy-selection-sizer.html)**
> - 🃏 **[7-Card 4:5 Visual Report (`enterprise-ai-gateway-token-quota-chargeback-apim-kong-litellm-envoy-selection-cards.html`)](../reports/enterprise-ai-gateway-token-quota-chargeback-apim-kong-litellm-envoy-selection-cards.html)**

---

## 1. 📌 Executive Summary

> **Bottom Line**: **Governing a shared 10-BU enterprise AI platform with legacy HTTP `RPM` rate limits and equal request-count billing (Tier 1) allows a single runaway Agent loop (`64K` prompt + `4K` `<think>` tokens = `420x` the compute of a short query) to exhaust the entire shared PTU/GPU pool, causing **`29.4%` collateral throttling** across sibling departments and **`38.5%` chargeback distortion**; meanwhile, static non-borrowable TPM slices with `max_tokens` pre-deduction (Tier 2) strand **`47.5%` of shared capacity**! Deploying **Tier 4 (Two-Phase 4-Tuple Weighted Token Bucket + Elastic Idle Quota Borrowing `<80%` + Per-BU Dedicated PAYGO Spillover `>=85%`)** cuts FinOps chargeback error to **`0.02%`**, eliminates noisy-neighbor drops (**`0.0%`**), boosts shared pool utilization to **`89.4%`**, and reduces net monthly spend by **`61.0%` (`$145,000 -> $56,500`)**!**

| Quota & Chargeback Tier (10 Business Units Sharing Pool) | Rate-Limiting & Quota Scheduling | Chargeback Error | Shared Pool Utilization | Noisy-Neighbor Drop | Monthly Net Spend | Architect Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **🟥 Tier 1: Legacy HTTP `RPM` + Per-Call Split** | Counts requests only; ignores token/CoT weight | **`38.5%` ✕ (Distorted)** | `48.0%` ✕ | **`29.4%` ✕ (Pool Starved)** | `$145,000 / mo` ✕ | **Never use raw RPM limits for LLMs** |
| **🟨 Tier 2: Static TPM Slices + `max_tokens` Lock** | Non-borrowable slices; locks full `max_tokens` | `16.2%` | `52.5%` ✕ (Stranded) | `1.2%` | `$118,000 / mo` | Phantom `max_tokens` locks starve tenants |
| **🟦 Tier 3: Two-Phase Streaming Bucket + `$ Cap`** | P80 soft reserve + post-SSE reconcile + USD caps | `0.8%` ✅ | `76.0%` ✅ | `0.3%` ✅ | `$78,000 / mo` ✅ | Solid enterprise baseline (APIM / LiteLLM) |
| **🟩 Tier 4: 4-Tuple Weighted Bucket + Elastic Borrowing + Per-BU Spillover (Recommended)** | **Separates cached/think tokens + idle borrowing + BU spillover** | **`0.02%` 🏆** | **`89.4%` 🏆** | **`0.0%` 🏆** | **`$56,500 / mo` 🏆 (`-61.0%`)** | **Enterprise CCoE FinOps gold standard** |

---

## 2. 🎯 Architect Deep-Dive Q&A: Two-Phase Token Buckets & Elastic Borrowing

### Q1: How does the Two-Phase Streaming Weighted Token Bucket work?
1. **Phase 1 (Pre-Flight Soft Reservation)**: Reserves $\Delta Q_{\text{reserve}} = N_{\text{in}} + 4 \times \min(\text{max\_tokens}, \hat{N}_{\text{out\_p80}})$ upon request arrival.
2. **Phase 2 (Post-SSE Exact 4-Tuple Reconciliation)**: Reads `(Uncached_In, Cached_In, Reasoning_Tokens, Output_Tokens)` from the final SSE chunk (`include_usage: true`), computes $\Delta Q_{\text{actual}} = N_{\text{uncached}} + 0.5 N_{\text{cached}} + 4(N_{\text{think}} + N_{\text{out}})$, and immediately refunds $(\Delta Q_{\text{reserve}} - \Delta Q_{\text{actual}})$ back to the tenant's bucket!

### Q2: How does Hierarchical Elastic Quota Borrowing eliminate idle PTU/GPU waste?
- Each BU holds a guaranteed base TPM share. While aggregate pool utilization is `< 80%`, any BU can burst into idle pool capacity for free; once aggregate utilization reaches `>= 85%`, borrowing is instantaneously reclaimed and excess traffic from over-quota BUs is routed to that specific BU's own `Serverless PAYGO` billing key.

---

## 3. 🧭 Two-Phase Token Bucket & Elastic Spillover Flow

```mermaid
flowchart LR
    Req["BU Request (JWT dept_id)"] --> P1["Phase 1 Soft Reserve\n(N_in + 4 * P80_Out)"]
    P1 --> Gate{"Within BU Base Quota\nOR Pool Util < 80%?"}
    Gate -->|"Yes"| Shared["Shared PTU / K8s GPU Pool\n(89.4% Utilization)"]
    Gate -->|"No (Pool >= 85%)"| Spill["Spillover to BU's Own PAYGO Key\n(0% Collateral Impact)"]
    Shared & Spill --> P2["Phase 2 Post-SSE Reconcile\n4-Tuple Chargeback + Refund 🏆"]
```

---

## 4. 📊 AI Gateway Quota & FinOps Capability Matrix

| Dimension | `Azure APIM (AI Gateway)` | `LiteLLM Enterprise` | `Kong AI Gateway` | `Envoy AI Gateway (K8s GIE)` |
| :--- | :--- | :--- | :--- | :--- |
| **Streaming Token Limit** | **Native `llm-token-limit` Policy** 🏆 | Virtual Key RPM/TPM/Budget | **`AI Rate Limiting Advanced`** 🏆 | **Native CEL Token Rate Limit** 🏆 |
| **Real-Time USD Hard Cap** | Azure Cost Alerts | **Built-in Team/Key/User `$` Hard Cap** 🏆 | Plugin-extensible | External FinOps Controller |
| **4-Tuple Token Accounting** | `llm-emit-token-metric` | **Native Cached + Reasoning Cost Split** 🏆 | Multi-Cloud Usage Extractor | OpenTelemetry GenAI Spans |

---

## 5. 🛠️ Production Readiness Checklist

- [ ] **Inject `stream_options: {"include_usage": true}` at the Gateway**: Guarantee every streaming SSE response emits exact 4-tuple token counts for Phase-2 reconciliation.
- [ ] **Weight Output & `<think>` Tokens at `4x–5x` Input Tokens in Quota Buckets**: Align rate-limiting math with physical GPU HBM decode time and cloud PTU weights.
