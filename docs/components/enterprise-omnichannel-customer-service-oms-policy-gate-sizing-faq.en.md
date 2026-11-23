# Enterprise Omnichannel Customer Service & Order-Action Agent (`10M Tickets/Month`): `Legacy IVR + Human` vs. `Unconstrained 70B` vs. `8B/70B Cascade + Deterministic OMS/OPA Refund Policy Gate` (`#556`)

> **Language / 语言**: [简体中文](./enterprise-omnichannel-customer-service-oms-policy-gate-sizing-faq.md) | **English** · [📱 Interactive H5 Study Lab](../h5/omnichannel-customer-service-oms-policy-gate-lab.html) · [🧮 Interactive Sizer](../reports/enterprise-omnichannel-customer-service-oms-policy-gate-sizing-sizer.html) · [🃏 7-Card Deck](../reports/enterprise-omnichannel-customer-service-oms-policy-gate-sizing-cards.html) · [Back to Catalog](../full-catalog-en.md)

![Architecture Summary](../../assets/svg/enterprise-omnichannel-customer-service-oms-policy-gate-sizing-faq-summary.svg)

---

## 1. Executive Summary (3-Minute Read)

1. **Why Unconstrained `70B` Customer Service Agents Suffer `8.4%` Prompt-Injected Refund Abuse While Legacy `IVR + Human` Costs `$6.80 / Ticket`**:
   - Connecting a `70B LLM` directly to Order Management System (`OMS`) refund APIs leads to **`8.4%` unauthorized goodwill/refund payouts** under adversarial customer prompting and requires **64 H200 GPUs (`$0.420 / ticket`)**.
2. **The `78% 8B FP8 / 22% 70B Cascade + Deterministic OMS/OPA Refund Policy Gate` Architecture**:
   - Routes `78%` routine tracking/FAQ tickets to an `8B FP8` pool, escalates `22%` multi-turn return/dispute negotiations to `70B`, and validates every refund/credit action through a deterministic **OMS + OPA Policy Engine** (`delivery_days <= 7 && refund_amount <= paid_amount`) — achieving **`88.6%` automated containment**, **`0.0%` unauthorized refund leakage**, and **`$0.038 / ticket` (`-99.4%` vs. human, `-91.0%` vs. unconstrained 70B)** on **12x H200 GPUs**!

---

## 2. Quantitative Architecture Decision Matrix (`10M Tickets/Month`)

| Metric | Scheme A: Legacy `IVR` + Human Agents | Scheme B: Unconstrained `70B` Direct API | Scheme C: `8B/70B Cascade + OMS/OPA Gate` |
| :--- | :--- | :--- | :--- |
| **Automated Containment Rate** | `18.5%` | `74.0%` | **`88.6%` (`+70.1 pts` vs. IVR)** |
| **Unauthorized Refund Leakage** | `0.4%` | `8.4%` (Abuse) | **`0.0%` (Deterministic OPA Gate)** |
| **Required H200 GPUs** | `0 GPUs` | `64 GPUs` | **`12 GPUs` (`-81.3%` GPUs)** |
| **Cost per Ticket (`$/Ticket`)** | `$6.800` | `$0.420` | **`$0.038` (`-91.0%` vs. 70B)** |
