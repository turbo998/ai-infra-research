# Enterprise Consumer Packaged Goods (`CPG`) Revenue Growth Management (`RGM`), Trade Promotion Optimization (`TPO`), True Net-Lift Decomposition (`Forward-Buy / Pantry-Dip / Cannibalization`) & Retailer `EDI 812` Deduction Audit Agent Architecture Selection Guide: Roll-Forward Excel Calendars + Unaudited Deductions vs. Prompt-Only Promo LLM vs. Single-SKU Elasticity ML vs. Hierarchical Bayesian Net-Lift + 52-Week `JBP MILP` + `EDI 812 / POS` Clawback Agent

> **Language / 语言**: **English (Current)** | [中文](./enterprise-cpg-trade-promotion-optimization-lift-deductions-milp-selection-faq.md)  
> **Scope**: End-to-end production reference architecture and TCO selection guide for Cloud & AI Solution Architects, CPG Chief Commercial Officers & Heads of Revenue Growth Management (`RGM`), Trade Marketing Directors, and `TPM / Deductions` (`SAP TPM / Visualfabriq / HighRadius / o9`) Leads.  
> **Provenance**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`). Deterministic evaluation model sealed in [`experiments/enterprise_cpg_trade_promotion_optimization_lift_deductions_milp_selection/results.json`](../../experiments/enterprise_cpg_trade_promotion_optimization_lift_deductions_milp_selection/results.json).

![Enterprise CPG Trade Promotion Optimization, Lift Decomposition & EDI 812 Deductions Agent Summary](../../assets/svg/enterprise-cpg-trade-promotion-optimization-lift-deductions-milp-selection-faq-summary.svg)

---

## ① 📌 Executive Summary (3-Minute Architect Verdict)

Managing **`$180M/year` in trade promotion spend (`15%–25%` of gross revenue) across `420` CPG SKUs, `18` national retail banners, and `42,000` annual retailer invoice deduction claims (`EDI 812`)** exposes three massive profit leaks:
1. **Celebrating Promo-Week Factory Shipment Spikes as "True Lift" Leaves `58.4%` of Promotions Value-Destroying (`Negative Net ROI`)**: When a 24-pack beverage runs at `30% off`, factory shipments jump `+140%`—but `52%` is retailer warehouse **forward-buying**, `38%` is consumer **pantry-loading** (borrowed from the next 4 weeks' baseline), and `34%` is **cross-SKU cannibalization** of higher-margin 12-packs (`-28%` net margin ROI)!
2. **Auto-Writing Off Small Retailer `EDI 812` Chargeback Deductions (Tier 1) Leaks `11.6%` of Trade Spend to Unauthorized Over-Billings and Unbuilt Ghost Endcaps**.
3. **Un-Synchronized Promotional Calendars Trigger `14.2%` On-Shelf Availability (`OSA`) Stockouts When Overlapping Retail Banners Drain Factory Inventory Simultaneously**.

Evaluated across a **`$180M` annual trade-spend budget, 420 SKUs, 18 retail banners, and 42,000 annual deduction claims**, the four CPG RGM architectures compare as follows:

| Architectural Tier | Core Stack Combination | Negative-ROI Promo Share | Invalid Deduction Leakage | Promo Out-of-Stock (`OOS`) | Net Margin Forecast MAPE | Monthly Trade & Deduction Loss TCO | Production Verdict |
|---|---|---|---|---|---|---|---|
| **🟥 Tier 1: Roll-Forward Excel + Blind Write-Offs** | Copy last year's calendar + gross shipment lift + write off `<$10K` claims | `58.4%` ✕ *(over half lose money)* | `11.6%` ✕ *(unverified chargebacks)* | `14.2%` ✕ | `29.5%` ✕ | `$1,380,000 / mo` ✕ | ⛔ Subsidizes baseline volume & leaks deductions |
| **🟨 Tier 2: Prompt-Only Promo Calendar LLM** | Asking a general LLM to plan discounts & write dispute letters | `54.0%` ✕ | `13.2%` ✕ *(no POS scan proof)* | `15.8%` ✕ | `33.0%` ✕ | `$1,490,000 / mo` ✕ | ⛔ Blind to cross-elasticity & EDI 812 math |
| **🟦 Tier 3: Single-SKU Own-Price Elasticity ML** | Log-log single-SKU POS regression without pantry-dip or cannibalization | `24.5%` ✕ *(fooled by stockpiling)* | `5.4%` ✕ | `6.8%` ✕ | `11.8%` ✕ | `$510,000 / mo` ✕ | ⚠️ Ignores post-promo trough & sister-SKU halo |
| **🟩 Tier 4: Bayesian Net-Lift + `52-Wk JBP MILP` + `EDI 812` Agent** | **Hierarchical Own/Cross/Pantry-Dip Decomposition + `JBP/CPFR MILP` + Endcap `VLM` + `EDI 812 vs. POS` Clawback Agent** | **`3.2%` 🏆** *(-55.2 pts)* | **`0.2%` 🏆** *(-11.4 pts)* | **`0.9%` 🏆** *(-13.3 pts)* | **`2.4%` 🏆** *(-27.1 pts)* | **`$115,000 / mo` 🏆** *(-91.7%)* | 🌟 **Recommended CPG Revenue Growth Standard** |

> 💡 **Golden Rule for CPG Revenue Growth Management (`RGM`) & Trade AI Architects**: **"Never evaluate trade promotions on promo-week factory shipments—decompose gross lift via a Bayesian hierarchical model that subtracts retailer forward-buying, 4-week post-promo consumer pantry-loading dips, and cross-SKU cannibalization; optimize the 52-week Joint Business Plan (`JBP`) calendar via a `MILP` interlocked with `MAP` price floors and factory `CPFR` safety stock; and audit 100% of retailer `EDI 812` deductions by reconciling billed units against syndicated `POS` scan-downs and in-store endcap `VLM` proof-of-performance."**

---

## ② 🎯 Customer & Architect Deep-Dive Q&A (3 CPG Trade Promotion & Deduction Pain Points)

### Q1: How does True Net-Incremental Lift Decomposition expose that a `+140%` shipment spike actually lost `-28%` in net margin?
- **First-Principles Reality**:
  - Gross promo-week shipment lift conflates four distinct effects: (1) true category expansion / brand switching, (2) **retailer forward-buying** into DC warehouses, (3) **consumer pantry-loading** (stockpiling followed by a 4-week post-event trough), and (4) **sister-SKU cannibalization**.
  - Tier 4 computes **True Net Incremental Margin**:
    $$V_{i, t}^{\text{Net}} = V_{i, t}^{\text{Gross\_Lift}} - V_{i, t}^{\text{Forward\_Buy}} - \sum_{k=1}^{4} V_{i, t+k}^{\text{Pantry\_Dip}}, \quad \Delta \Pi_i = V_{i, t}^{\text{Net}} \cdot m_i^{\text{promo}} - \sum_{j \ne i} V_{j \to i}^{\text{Cannibal}} \cdot m_j^{\text{base}} - F_{\text{slot}}$$
  - Replacing value-destroying deep single-unit discounts (`-28%` ROI) with multi-buy thresholds + endcap displays (`+34%` ROI) cuts negative-ROI events from `58.4%` to **`3.2%`**.

### Q2: How does the `EDI 812` vs. `POS Scan` + Endcap `VLM` Auditor claw back `$47,200` on a single retailer deduction?
- **Three-Way Trade Spend Reconciliation (`Contract x POS Scans x Store VLM`)**:
  - When Retailer #4 deducts `$164,000` via `EDI 812` claiming `82,000` units sold under a `$2.00/unit` scan-down contract, Tier 4 cross-checks syndicated `Circana / NielsenIQ / EDI 852` store-level POS scan feeds (`58,400` actual promo scans) and field-rep **In-Store Endcap Planogram `VLMs`**—automatically assembling a line-item dispute package that recovers **`$47,200`** (`0.2%` leakage).

### Q3: How does the 52-Week `JBP + CPFR MILP` cut promo-induced out-of-stocks (`OOS`) from `14.2%` to `0.9%`?
- **Cross-Banner Staggering + Factory Lead-Time Synchronization**:
  - Enforces minimum 2-week staggering of hero-SKU promotions across major retail banners, 3-week post-promo price-recovery cool-downs, and 6-week advance `CPFR` factory production reservations (`0.9%` promo OOS).

---

## ③ 🧭 Visual Architecture Diagrams (Decision Tree + Net-Lift/JBP/Deduction Topology + Sequence Flow)

### 3.1 Architecture Selection Decision Tree

```mermaid
flowchart TD
    A["Enterprise CPG Trade Promotion Optimization (TPO) & Deductions<br/>($180M Annual Trade Spend + 420 SKUs + 18 Retail Banners + 42K Claims)"] --> B{"Rolls forward last year's Excel calendar + gross shipment lift<br/>and auto-writes off small retailer deductions?"}
    B -- "Yes (Roll-forward Excel or prompt-only LLM)" --> C["⛔ 58.4% Negative-ROI Promos & 11.6% Deduction Leakage (Tier 1 / Tier 2)!<br/>Rewards pantry-loading & writes off $1.38M/mo in wasted trade spend"]
    B -- "Uses price-elasticity ML" --> D{"Subtracts forward-buy, pantry-dip & cannibalization,<br/>and reconciles EDI 812 claims against POS scans & endcap VLMs?"}
    D -- "No (Single-SKU own-price regression only)" --> E["⚠️ Pantry-Dip & Cannibalization Blind Spot (Tier 3)<br/>24.5% negative-ROI promos & 5.4% unrecovered retailer deductions"]
    D -- "Yes (Bayesian Net-Lift + JBP MILP + EDI 812 Agent)" --> F["🟩 Tier 4: Bayesian Net-Lift + 52-Wk JBP MILP + EDI 812 Agent 🏆<br/>(3.2% Neg-ROI Share | 0.2% Deduction Leak | 0.9% Promo OOS | -91.7% Waste)"]
    style F fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#f8fafc
    style C fill:#7f1d1d,stroke:#ef4444,color:#fecaca
```

### 3.2 Tier 4 Bayesian Net-Lift Decomposition + 52-Week `JBP MILP` + `EDI 812` Audit Topology

```mermaid
flowchart LR
    subgraph Decomp["1. Bayesian 4-Way Net-Lift Engine"]
        Data["Factory Shipments + EDI 852<br/>+ Syndicated POS Scans"] --> Bayes["🔬 True Net-Incremental Decomposer<br/>Gross Lift - Forward-Buy<br/>- 4-Wk Pantry Dip - SKU Cannibalization"]
    end

    subgraph Calendar["2. 52-Week JBP & CPFR MILP"]
        Bayes --> MILP["🗓️ Cross-Banner Stagger & MAP MILP<br/>Price Floors + 3-Wk Cool-Down + CPFR Sync<br/>(3.2% Neg-ROI | 0.9% Promo OOS)"]
    end

    subgraph Audit["3. EDI 812 & Endcap VLM Clawback"]
        Claim["42,000 Annual EDI 812 Claims"] --> Match["🛡️ POS Scan × Contract × Endcap VLM<br/>Flags Over-Billed Scans & Ghost Displays<br/>Auto-Submits Dispute Package (0.2% Leak)"]
        Data & MILP --> Match
    end
```

### 3.3 End-to-End Sequence: Rejecting a `-28% ROI` Pantry-Loading Promo & Clawing Back `$47,200` on `EDI 812`

```mermaid
sequenceDiagram
    autonumber
    participant KAM as Key Account Manager (KAM) & Finance
    participant Bayes as Bayesian Net-Lift & 52-Wk JBP MILP
    participant Audit as EDI 812 × POS × Endcap VLM Auditor
    participant Portal as Retailer Supplier Dispute Portal

    KAM->>Bayes: Propose Q3 repeat of 30%-off 24-pack Sparkling Water (logged +140% shipment spike last year)
    Bayes->>Bayes: Decompose POS & shipment history: 52% forward-buy + 38% 4-wk pantry dip + 34% 12-pack cannibalization -> Net ROI = -28%!
    Bayes-->>KAM: ⚡ Reject deep discount! MILP substitutes "15% 2-for-$11 Multi-Buy + Endcap" (Net ROI = +34%, staggered from Banner #2)
    Portal->>Audit: Retailer submits EDI 812 Debit Memo #9081 deducting $164,000 ($2.00/unit on 82,000 claimed units)
    Audit->>Audit: Reconcile against store POS scan feed (58,400 actual units) + Endcap VLM -> isolate $47,200 over-deduction!
    Audit->>Portal: Auto-submit dispute package with store-by-store POS logs & contract Exhibit B -> recover $47,200!
```

---

## ④ 📊 4-Tier Architecture Decision Matrix

| Evaluation Dimension | 🟥 Tier 1: Roll-Forward Excel | 🟨 Tier 2: Prompt Calendar LLM | 🟦 Tier 3: Single-SKU Elasticity | 🟩 Tier 4: Net-Lift + `JBP MILP` + `EDI 812` |
|---|---|---|---|---|
| **Negative-ROI Promotion Share** | `58.4%` ✕ (subsidizes baseline) | `54.0%` ✕ | `24.5%` ✕ (ignores pantry/cannibal) | 🏆 **`3.2%` (4-way net-lift + 52-week `JBP MILP`)** |
| **Invalid Deduction Leakage** | `11.6%` ✕ (`<$10K` auto-write-off) | `13.2%` ✕ (unverified text letters) | `5.4%` ✕ | 🏆 **`0.2%` (`EDI 812 × POS Scan × Endcap VLM` clawback)** |
| **Promo Out-of-Stock (`OOS`) Rate** | `14.2%` ✕ (overlapping banner spikes) | `15.8%` ✕ | `6.8%` ✕ | 🏆 **`0.9%` (cross-banner staggering + `CPFR` stock sync)** |
| **Net Margin Forecast MAPE** | `29.5%` ✕ | `33.0%` ✕ | `11.8%` ✕ | 🏆 **`2.4%` (Bayesian hierarchical cross-elasticity tree)** |
| **Monthly Trade & Deduction Loss** | `$1,380,000 / mo` | `$1,490,000 / mo` | `$510,000 / mo` | **`$115,000 / mo` 🏆 (-91.7%)** |

---

## ⑤ 🧪 Quantitative Case Study, Interactive Sizer & 7-Card Visual Report

- **Interactive HTML Sizer**: 👉 [Open Enterprise CPG Trade Promotion Optimization (TPO) & Deductions Sizer](../reports/enterprise-cpg-trade-promotion-optimization-lift-deductions-milp-sizer.html)
- **7-Card 4:5 Visual Benchmark Report**: 👉 [Open 7-Card Visual Benchmark Deck](../reports/enterprise-cpg-trade-promotion-optimization-lift-deductions-milp-selection-cards.html)
- **Deterministic Verification**:
  ```bash
  python experiments/enterprise_cpg_trade_promotion_optimization_lift_deductions_milp_selection/run.py --check
  python -m unittest tests/test_enterprise_cpg_trade_promotion_optimization_lift_deductions_milp_selection.py
  ```

---

## ⑥ 🛠️ Production Readiness & Architecture Review Checklist

- [ ] **Evaluate Promotions Strictly on Net Incremental Margin Rather Than Gross Promo-Week Factory Shipments**: Subtract retailer forward-buying, 4-week post-promo consumer pantry-loading dips, and cross-SKU cannibalization.
- [ ] **Solve 52-Week Joint Business Plan (`JBP`) Calendars via a Constrained `MILP`**: Enforce Minimum Advertised Price (`MAP`) floors, 3-week post-discount cool-down windows, cross-banner staggering, and factory `CPFR` lead-time locks.
- [ ] **Eliminate Auto-Write-Off Thresholds on Retailer `EDI 812` Deductions**: Automatically reconcile 100% of debit memos against store-level `POS / EDI 852` scan-down quantities and contracted allowance rates.
- [ ] **Gate Fixed Slotting and Endcap Display Allowances on Field `VLM` Proof-of-Performance**: Verify planogram facings, endcap erection, and promo price tags from store-visit photos.
- [ ] **Include Rival Brand Cross-Price Elasticity Terms in the Hierarchical Model**: Separate competitor counter-promotion shocks from own-brand promotional performance.
