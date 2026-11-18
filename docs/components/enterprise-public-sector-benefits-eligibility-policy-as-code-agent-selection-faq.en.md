# Enterprise Sovereign & Public-Sector Benefits Eligibility (`SNAP / Medicaid / Unemployment / Tax Credit`), Statutory Policy-as-Code (`OpenFisca / Catala`) Compiler, Household Income/Asset Cross-Verification & Administrative Appeal Audit Agent Architecture Selection Guide: Manual Caseworker Queue + 42-Day Backlog vs. Prompt-Only Statute LLM vs. Policy Manual Vector RAG vs. Multimodal Fact Extractor + `Catala / OpenFisca` Policy-as-Code `AST` + Wage Cross-Match + `Goldberg v. Kelly` Proof-Tree Agent

> **Language / 语言**: **English (Current)** | [中文](./enterprise-public-sector-benefits-eligibility-policy-as-code-agent-selection-faq.md)  
> **Scope**: End-to-end production reference architecture and TCO selection guide for Cloud & AI Solution Architects, State/Federal Chief Information Officers (`CIOs`), Integrated Eligibility System (`IES`) Engineering Leads, and Digital Government Policy-as-Code Teams.  
> **Provenance**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`). Deterministic evaluation model sealed in [`experiments/enterprise_public_sector_benefits_eligibility_policy_as_code_agent_selection/results.json`](../../experiments/enterprise_public_sector_benefits_eligibility_policy_as_code_agent_selection/results.json).

![Enterprise Public-Sector Benefits Eligibility & Policy-as-Code Agent Summary](../../assets/svg/enterprise-public-sector-benefits-eligibility-policy-as-code-agent-selection-faq-summary.svg)

---

## ① 📌 Executive Summary (3-Minute Architect Verdict)

Administering **220,000 monthly household benefit applications and recertifications (`SNAP / Medicaid / TANF / Unemployment / Childcare Subsidies`) across `8,400` statutory subclauses and `$145M/month` in public disbursements** exposes three critical legal and operational failure modes:
1. **Prompt-Only Statute LLMs Commit `24.2%` Deduction Math Errors & Wrongfully Deny Eligible Seniors (`19.6%` Improper Rate)**: Public-benefit statutes are deeply nested default-with-exception trees. Under **`7 CFR § 273.9`**, households generally must pass a `130%` Federal Poverty Level (`FPL`) gross-income test—*unless* the household includes an elderly (`>=60`) or disabled member, which **waives the `130%` gross cap** and unlocks uncapped excess shelter and medical (`>$35`) deductions! Feeding policy manuals into a prompt-only LLM (Tier 2) or chunked vector RAG (Tier 3) retrieves the general `130%` rule from Page 12 while missing the elderly exemption on Page 48—wrongfully denying vulnerable seniors and triggering a **`39.0%` administrative appeal reversal rate**!
2. **Pay-Period Conversion Errors (`2.15×` Bi-Weekly vs. `2.0×` Semi-Monthly) Drive Federal Improper Payment Penalties (`9.8%` Manual Math Error)**: Multiplying a `$1,120` bi-weekly paystub (26 paychecks/year) by `2` instead of the statutory **`2.15` conversion factor (`$2,408.00`)** miscalculates every allotment.
3. **Opaque Adverse Action Notices Violate Constitutional Due Process (*Goldberg v. Kelly*)**: Issuing benefit reductions without a line-by-line calculation worksheet and exact statutory citations causes **`28.5%` automatic reversals** at fair hearings.

Evaluated across **220,000 monthly household determinations, 8,400 statutory rules, and `$145M/month` in disbursements**, the four architectural tiers compare as follows:

| Architectural Tier | Core Stack Combination | Improper Payment & Wrongful Denial Rate | Statutory Deduction Math Error | Appeal Hearing Reversal Rate | Determination Turnaround Lag | Monthly Improper-Pay & Admin TCO | Production Verdict |
|---|---|---|---|---|---|---|---|
| **🟥 Tier 1: Manual Caseworker Queue (42d Backlog)** | Manual paystub math + legacy mainframe tables + opaque adverse action letters | `11.4%` ✕ | `9.8%` ✕ | `28.5%` ✕ *(no math worksheet)* | `42.0 Days` ✕ | `$920,000 / mo` ✕ | ⛔ 6-week backlog & high error rate |
| **🟨 Tier 2: Prompt-Only Statute LLM Calculator** | Feeding policy PDFs & paystubs into an LLM to decide eligibility and allotment in text | `19.6%` ✕ *(misses senior waiver)* | `24.2%` ✕ *(fails deduction caps)* | `39.0%` ✕ | `0.5 Days` | `$1,240,000 / mo` ✕ | ⛔ Violates statutory determinism & due process |
| **🟦 Tier 3: Policy Manual Vector RAG Only** | Embedding chunks of policy manuals without a deterministic Policy-as-Code compiler | `6.9%` ✕ | `7.5%` ✕ | `16.4%` ✕ | `2.0 Days` | `$440,000 / mo` ✕ | ⚠️ Splits general rules from cross-page exemptions |
| **🟩 Tier 4: Multimodal Fact Extractor + `Catala/OpenFisca` `AST` + Proof-Tree Agent** | **VLM Fact Extractor + State Wage Cross-Match + `Catala / OpenFisca` Legislative `AST` + `Goldberg v. Kelly` Proof-Tree Notice** | **`0.2%` 🏆** *(-11.2 pts)* | **`0.0%` 🏆** *(Exact math)* | **`0.5%` 🏆** *(-28.0 pts)* | **`0.1 Days (2.4h)` 🏆** | **`$82,000 / mo` 🏆** *(-91.1%)* | 🌟 **Recommended Architecture** |

> 💡 **Golden Rule for Public-Sector Eligibility & Digital Government Architects**: **"Strictly separate document fact extraction from legal rule evaluation—use multimodal VLMs only to extract structured household facts (with confidence gates to human caseworkers), execute all eligibility and benefit waterfall calculations through a deterministic `Catala / OpenFisca` Policy-as-Code compiler (`0.0%` math error), and attach a complete `Goldberg v. Kelly` mathematical proof trace and statutory citation to every citizen notice."**

---

## ② 🎯 Customer & Architect Deep-Dive Q&A (3 Real-World Public-Sector Eligibility Pain Points)

### Q1: Why does vector RAG over policy manuals wrongfully deny eligible elderly households, and how does `Catala / OpenFisca` Policy-as-Code solve it?
- **First-Principles Reality**:
  1. **Statutes Are Non-Monotonic Default-Logic Trees, Not Flat Paragraphs**: In `7 CFR § 273.9`, the base rule imposes a `130% FPL` gross-income ceiling, while Sub-Clause `(b)` overrides that ceiling for households containing an elderly (`>=60`) or disabled member and unlocks uncapped excess shelter deductions. Vector similarity search retrieves the high-similarity base rule while dropping distant exception clauses.
  2. **Literate Legislative Programming (`Catala / OpenFisca`)**: Tier 4 compiles statutory text side-by-side with formally verified **`Catala / OpenFisca` code**. The VLM extracts structured facts (`{members: [{age: 67}], biweekly_pay: 1120, medical_oop: 480, rent: 1350}`), and the deterministic compiler evaluates every cross-referenced exemption and deduction cap with **`0.0%` arithmetic or logic drift**.

### Q2: How does Tier 4 eliminate the `9.8%` paystub conversion error while protecting working families from multi-program "Benefit Cliffs"?
- **Statutory Pay-Cadence Normalization + Multi-Program Simulation**:
  - **Bi-Weekly (`2.15×`) vs. Semi-Monthly (`2.0×`) Verification**: Tier 4 inspects pay-period start/end dates on uploaded paystubs and cross-matches quarterly state Unemployment Insurance (`UI`) wage tapes—applying the exact statutory `2.15` multiplier (`$1,120 × 2.15 = $2,408.00/mo`) while flagging synthetic-identity fraud rings sharing bank routing numbers across 180 burner applications.
  - **Cross-Program Benefit-Cliff Simulator**: Evaluates `SNAP + Medicaid + Childcare + TANF` simultaneously in `OpenFisca` so caseworkers and citizens can see exact net household resources before accepting a shift or wage change.

### Q3: How does the `Goldberg v. Kelly` Proof-Tree Generator slash administrative hearing reversals from `28.5%` to `0.5%`?
- **Deterministic Calculation Worksheets + Exact `CFR` Citations**:
  - Every `Catala / OpenFisca` evaluation emits a cryptographically hashed **Execution Proof Tree** that renders a human-readable line-by-line worksheet (`Gross $2,408.00 - 20% Earned Income Deduction $481.60 - Standard Deduction $238.00 - Excess Medical $445.00 - Uncapped Excess Shelter $300.80 = Net Countable Income $942.60 -> Monthly Allotment $512.00` under `7 CFR § 273.9(d)(3) & § 273.10(e)`).

---

## ③ 🧭 Visual Architecture Diagrams (Decision Tree + Policy-as-Code Topology + Sequence Flow)

### 3.1 Architecture Selection Decision Tree

```mermaid
flowchart TD
    A["Enterprise Public-Sector Benefits Eligibility Platform<br/>(220,000 Monthly Applications + 8,400 Statutory Rules + $145M/Mo Disbursed)"] --> B{"Uses manual 42-day paperwork queues<br/>or prompt-only LLM free-text benefit math?"}
    B -- "Yes (Manual backlog or LLM text math)" --> C["⛔ Improper Payments & Due-Process Reversals (Tier 1 / Tier 2)!<br/>11.4%–19.6% improper rate, 24.2% math errors & 39% appeal reversals"]
    B -- "Uses automated policy retrieval" --> D{"Compiles statutes into deterministic Catala/OpenFisca AST<br/>with Goldberg v. Kelly proof-tree notices?"}
    D -- "No (Policy Manual Vector RAG only)" --> E["⚠️ Missed Statutory Exemptions (Tier 3)<br/>6.9% improper denials on elderly/disabled cross-page exceptions"]
    D -- "Yes (VLM Extractor + Catala/OpenFisca AST + Proof Tree)" --> F["🟩 Tier 4: Multimodal Fact Extractor + Catala/OpenFisca AST + Proof-Tree Agent 🏆<br/>(0.2% Improper Rate | 0.0% Math Error | 2.4h Turnaround | 0.5% Reversal)"]
    style F fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#f8fafc
    style C fill:#7f1d1d,stroke:#ef4444,color:#fecaca
```

### 3.2 Tier 4 Multimodal Fact Extractor + `Catala / OpenFisca` `AST` + Due-Process Proof-Tree Topology

```mermaid
flowchart LR
    subgraph Extract["1. Fact Extraction & Wage Cross-Match Plane"]
        Docs["📱 Smartphone Paystubs/Leases/Medical<br/>(Low-Confidence Handwriting -> Human)"] --> VLM["👁️ Multimodal Fact Extractor<br/>Detects Bi-Weekly (2.15x) Cadence<br/>Outputs Structured JSON Facts Only"]
        Wage["🏛️ State UI Wage Tape + Bank Graph"] --> Cross["🔍 Income Reconciler & Fraud Ring Gate<br/>Blocks 180-App Synthetic Identity Rings"]
        VLM --> Cross
    end

    subgraph Law["2. Policy-as-Code Deterministic Compiler"]
        Cross --> Catala["⚖️ Catala / OpenFisca Legislative AST<br/>8,400 Versioned Statutory Rules<br/>7 CFR 273.9 Senior Waivers (0.0% Math Error)"]
    end

    subgraph Notice["3. Due-Process Proof Tree & Cliff Simulator"]
        Catala --> Proof["📜 Goldberg v. Kelly Notice Generator<br/>Line-by-Line Worksheet + CFR Citations<br/>(0.5% Hearing Reversal Rate)"]
        Catala --> Cliff["📈 Multi-Program Benefit-Cliff Simulator"]
    end
```

### 3.3 End-to-End Sequence: Evaluating Household `#HH-77412` in 2.4 Hours with Zero Statutory Error

```mermaid
sequenceDiagram
    autonumber
    participant HH as Applicant Household #HH-77412 (67yo Senior)
    participant VLM as Multimodal Fact Extractor & Wage Matcher
    participant Catala as Catala / OpenFisca Engine (7 CFR 273.9)
    participant Notice as Due-Process Proof-Tree Generator

    HH->>VLM: Upload 2 bi-weekly $1,120 paystubs, $480 medical receipts & $1,350 rent lease (gross income = 142% FPL)
    VLM->>VLM: Verify Bi-Weekly (26/yr) cadence against state UI wage tape; emit structured household fact JSON
    VLM->>Catala: Pass facts (3 members, max_age=67, biweekly_pay=$1,120, medical=$480, rent=$1,350)
    Catala->>Catala: Trigger 7 CFR 273.9(b) elderly exemption (waive 130% gross test); compute $2,408 gross -> $942.60 net income
    Catala->>Notice: Output deterministic allotment = $512.00/mo with SHA-256 execution proof tree (0.0% math error)
    Notice-->>HH: ✅ Deliver approval + line-by-line calculation worksheet + exact CFR citations in 2.4 hours!
```

---

## ④ 📊 4-Tier Architecture Decision Matrix

| Evaluation Dimension | 🟥 Tier 1: Manual Queue (42d) | 🟨 Tier 2: Prompt Statute LLM | 🟦 Tier 3: Policy Vector RAG | 🟩 Tier 4: VLM + `Catala/OpenFisca AST` |
|---|---|---|---|---|
| **Improper Payment / Wrongful Denial** | `11.4%` ✕ (manual fatigue slips) | `19.6%` ✕ (misses senior waivers) | `6.9%` ✕ (chunking drops exceptions) | 🏆 **`0.2%` (decoupled VLM facts + legislative `AST`)** |
| **Statutory Deduction Math Error** | `9.8%` ✕ (misses 2.15× factor) | `24.2%` ✕ (free-text math hallucinations) | `7.5%` ✕ | 🏆 **`0.0%` (compiled `Catala / OpenFisca` waterfall)** |
| **Appeal Hearing Reversal Rate** | `28.5%` ✕ (opaque notice letters) | `39.0%` ✕ (unexplainable black box) | `16.4%` ✕ | 🏆 **`0.5%` (`Goldberg v. Kelly` line-by-line worksheet)** |
| **Determination Turnaround Time** | `42.0 Days` ✕ (severe backlog) | `0.5 Days` | `2.0 Days` | 🏆 **`0.1 Days (2.4 hours end-to-end)`** |
| **Monthly Improper-Pay & Admin TCO** | `$920,000 / mo` | `$1,240,000 / mo` | `$440,000 / mo` | **`$82,000 / mo` 🏆 (-91.1%)** |

---

## ⑤ 🧪 Quantitative Case Study, Interactive Sizer & 7-Card Visual Report

- **Interactive HTML Sizer**: 👉 [Open Enterprise Public-Sector Benefits Eligibility & Policy-as-Code Sizer](../reports/enterprise-public-sector-benefits-eligibility-policy-as-code-agent-sizer.html)
- **7-Card 4:5 Visual Benchmark Report**: 👉 [Open 7-Card Visual Benchmark Deck](../reports/enterprise-public-sector-benefits-eligibility-policy-as-code-agent-selection-cards.html)
- **Deterministic Verification**:
  ```bash
  python experiments/enterprise_public_sector_benefits_eligibility_policy_as_code_agent_selection/run.py --check
  python -m unittest tests/test_enterprise_public_sector_benefits_eligibility_policy_as_code_agent_selection.py
  ```

---

## ⑥ 🛠️ Production Readiness & Architecture Review Checklist

- [ ] **Never Allow an LLM to Compute Statutory Deductions or Decide Benefit Eligibility in Free Text**: Use LLMs/VLMs strictly to extract structured household facts into a typed schema evaluated by **`Catala / OpenFisca`**.
- [ ] **Version All Statutory Parameters (`FPL`, `SUA`, Deduction Caps) in an Effective-Date Parameter Tree**: Ensure historical audits and October 1 cost-of-living rollovers resolve deterministically by application timestamp.
- [ ] **Route Low-Confidence Handwritten Receipts to a Side-by-Side Caseworker Verification Queue**: Never silently guess ambiguous income or rent digits.
- [ ] **Attach a Complete `Goldberg v. Kelly` Line-by-Line Calculation Worksheet and Exact Statutory Citation to Every Citizen Notice**: Guarantee procedural due process and eliminate administrative hearing reversals.
- [ ] **Cross-Match Extracted Income Against State Wage/Tax Tapes and Screen Disbursement Accounts via Bipartite Fraud Graphs**: Stop synthetic-identity rings without adding friction for genuine families.
