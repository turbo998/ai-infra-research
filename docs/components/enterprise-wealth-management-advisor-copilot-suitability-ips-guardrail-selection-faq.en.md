# Enterprise Wealth Management & Private Banking Advisor Copilot (`SEC Reg BI / FINRA Rule 2111 / MiFID II`), Client Investment Policy Statement (`IPS`) Suitability Solver, Approved CIO House-View Grounding & Promissory-Language Guardrail Selection Guide

> **Target Audience**: Cloud & AI Solution Architects, WealthTech & Private Banking Engineering Leaders, and Chief Compliance Officers building Financial Advisor (`FA`) Copilots across `BlackRock Aladdin Wealth`, `Salesforce Financial Services Cloud (FSC)`, `Charles River`, and `Avaloq` under **`SEC Regulation Best Interest (Reg BI)`**, **`FINRA Rules 2111 (Suitability) & 2210 (Communications with the Public)`**, and **`EU MiFID II`**.
> **Core Question**: When a wealth management institution empowers **`12,000` Financial Advisors managing `$850B` AUM across `480,000` high-net-worth households (`360,000` monthly portfolio reviews and client briefs)**, why do naive "Unconstrained LLM Ticker Advice & Freeform Email Drafting (Tier 1)" or "CIO Research PDF RAG Without Deterministic Client `IPS` Solvers (Tier 3)" result in a **`16.8%` `Reg BI / MiFID II` suitability violation rate, `11.4%` `FINRA Rule 2210` promissory-language breaches (`"guaranteed yield"`), and `$720,000/month` in supervisory rework and compliance remediation TCO**? How does a **4-stage architecture combining a Deterministic Client `IPS` + Approved Product List (`APL`) Convex Portfolio Solver (`Aladdin / CVXPY`) + Versioned `CIO House-View` Narrative Blocks + Pre-Send `FINRA 2210` Promissory Linter & Mandatory Disclosure Injector + `SEC Rule 17a-4` WORM Archiving** achieve **`0.0%` suitability and promissory violations**, compress advisor meeting prep from `52.0 minutes` to **`3.5 minutes`**, and reduce monthly TCO by **`89.2%` (`$720,000 -> $78,000/month`)**?

![Enterprise Wealth Management Advisor Copilot, IPS Suitability Solver & FINRA 2210 Guardrail Summary](../../assets/svg/enterprise-wealth-management-advisor-copilot-suitability-ips-guardrail-selection-faq-summary.svg)

---

## 1. 📌 Executive Summary (3-Minute Verdict)

In regulated wealth management and private banking, **a generative LLM must never act as a freewheeling stock picker—selecting securities and sizing tax-aware allocations belongs strictly to a deterministic convex portfolio solver (`Client IPS + Approved Product List`), while the LLM's sole role is translating those mathematically verified trades and pre-approved Chief Investment Officer (`CIO House-View`) blocks into client-ready prose**:

| Architecture Tier | Best-Fit Scenario | `Reg BI / MiFID II` Suitability Breach | `FINRA 2210` Promissory Breach | Advisor Meeting Prep Time | Monthly Compliance & Advisor Prep TCO | Architect Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Tier 1: Unconstrained LLM Ticker Advice & Freeform Email Drafting (Anti-Pattern)** | ❌ **Never use in regulated broker-dealer or RIA workflows** | **`16.8%` (off-APL / risk breach!)** | **`11.4%` (`"guaranteed"` claims)** | `12.0 min` | `$720,000 / mo` | ❌ **Reject**: Severe `SEC Reg BI`, `FINRA 2210` & `MiFID II` enforcement liability |
| **Tier 2: Legacy 100% Manual Advisor Prep + `Series 24` Principal Queue** | Bespoke ultra-high-net-worth family office restructuring | `0.4%` | `0.6%` | **`52.0 min` (`2.8-day` queue!)** | `$540,000 / mo` | ❌ **Reject**: Takes `2.8 days` to approve client emails after a Fed rate move |
| **Tier 3: CIO Research PDF RAG Only (No Deterministic `IPS` or Tax-Lot Solver)** | Internal morning advisor macro news digest | `8.2%` (concentration/tax breach) | `2.8%` | `9.5 min` | `$310,000 / mo` | ⚠️ **Fragile**: Summarizes macro views well, but breaches client `IPS` caps |
| **Tier 4: Deterministic `IPS/APL` Convex Solver + `CIO House-View` + `FINRA 2210` Gate (Recommended)** | **Enterprise Private Banking & Wealth Management Production Standard** | **`0.0%` (100% `IPS/APL` bound)** | **`0.0%` (linted + disclosures)** | **`3.5 min` (`15x` faster)** | **`$78,000 / mo` (`-89.2%`)** | 🏆 **Production Gold Standard**: Solver picks trades; CIO grounds macro; 2210 gates prose |

### 💡 Three Golden Rules for Wealth Management AI Architects
1. **Never Let an LLM Pick Tickers or Compute Portfolio Weights: Delegate All Allocation & Tax-Lot Math to a Deterministic `IPS + APL` Convex Solver (`Aladdin / CVXPY`)**:
   * Every wealth client account is governed by a binding **Investment Policy Statement (`IPS`)** specifying risk tolerance (`Conservative / Moderate / Growth`), liquidity floor (`>=10%`), single-issuer concentration cap (`<=5%`), ESG exclusions (`No_Tobacco`, `No_Cluster_Munitions`), and tax-lot state (`30-day Wash-Sale` and short-term capital gains avoidance).
   * Candidate trades are computed deterministically over the bank's **`2,400-security Approved Product List (APL)`** via a convex optimizer—guaranteeing **`0.0%` `Reg BI / MiFID II` suitability breaches**.
2. **Ground Client Narratives Strictly in Versioned, Compliance-Approved `CIO House-View` Building Blocks (`3.5-Minute` Prep Time Without `Series 24` Queue Bottlenecks)**:
   * Freeform financial commentary requires manual `FINRA Series 24` principal supervisory review (`2.8-day` turnaround).
   * By restricting the LLM to assembling pre-approved **Chief Investment Officer (`CIO`) House-View Narrative Blocks** (tagged with `status = ACTIVE` and expiration `TTL`) around the client's deterministic `IPS` rebalance table, generated commentary qualifies for automated supervisory clearance in **`3.5 minutes`**.
3. **Enforce a Pre-Send `<45ms` `FINRA Rule 2210` Promissory-Language Linter, Mandatory Asset-Class Disclosure Injector, and `SEC Rule 17a-4` WORM Audit Trail**: Blocks promissory or unbalanced claims (`"guaranteed yield"`, `"risk-free upside"`, `"can't lose"`), appends mandatory product risk disclaimers (municipal bonds, private credit, options), and writes the immutable `(Client_IPS_Snapshot, Solver_Trades, CIO_Block_IDs, Final_Text)` bundle to **`SEC 17a-4` WORM storage**.

---

## 2. 🎯 Customer & Architect Critical Q&A

### Q1: Why does RAG over the bank's 500 CIO research PDFs (Tier 3) still cause an `8.2%` `Reg BI / MiFID II` suitability violation rate?
* **Macro Research Relevance Does Not Equal Account-Level Suitability**:
  * Even if the CIO's weekly report recommends overweighting semiconductor equities, recommending a semiconductor ETF to a client whose trust account already holds `38%` concentrated tech stock (above their `25%` `IPS` sector cap) or whose sale of existing bonds would trigger a massive short-term capital gains tax bill violates `SEC Reg BI`. Only a deterministic **`IPS + Tax-Lot` portfolio solver** can evaluate account-level constraints before the LLM drafts a single word.

### Q2: What happens if a rogue or hurried Financial Advisor prompts the Copilot: *"Write a persuasive email promising the client a safe, guaranteed 9% return on this structured note"*?
* **Dual Input/Output `FINRA Rule 2210` Guardrail**:
  * Regardless of advisor prompting, the output passes through a `<45ms` **`FINRA 2210 / MiFID II` Fair-and-Balanced Classifier & Regex/NLI Gate** that blocks promissory modifiers (`"guaranteed"`, `"assured"`, `"safe 9%"`), enforces net-of-fees return presentation, and injects mandatory structured-note principal-at-risk disclosures before the text can be copied or emailed.

---

## 3. 🧭 Visual Diagrams (Mermaid Decision Tree + Reference Architecture + Sequence Diagram)

### 3.1 Enterprise Wealth Management Advisor Copilot Decision Tree

```mermaid
flowchart TD
    A["Enterprise Wealth Management Advisor Copilot Request"] --> B{"Involves portfolio rebalancing, product recommendation, or client outreach?"}
    B -- "Yes (Regulated under SEC Reg BI, FINRA 2111/2210 & MiFID II)" --> C["Prohibit Freeform LLM Ticker Picking! Enforce Tier 4 Architecture"]
    C --> D["Stage 1: Deterministic IPS + APL + Tax-Lot Convex Solver (Aladdin / CVXPY)\n(Enforces Risk Tier, 2,400-Ticker APL, 5% Issuer Cap, ESG & Wash-Sale Rules)"]
    D --> E["Stage 2: Retrieve Active Pre-Approved CIO House-View Narrative Blocks (TTL-Verified)"]
    E --> F["Stage 3: Draft Personalized Client Portfolio Review & Trade Rationale"]
    F --> G["Stage 4: Pre-Send FINRA 2210 Promissory Linter + Mandatory Asset Disclaimers + SEC 17a-4 WORM Log"]
```

### 3.2 Tier 4 `IPS/APL` Convex Solver + `CIO House-View` + `FINRA 2210` Reference Architecture

```mermaid
flowchart LR
    subgraph Solver["🧮 1. Deterministic Suitability & Tax-Lot Solver (0.0% Breach)"]
        HOLD["Client Holdings + Tax Lots\n+ Binding IPS Constraints"] --> CVX["Aladdin / CVXPY Convex Optimizer\n(Strictly Bounded to 2,400-Ticker APL)"]
        APL["Approved Product List (APL)\nRisk Ratings R1..R5"] --> CVX
    end

    subgraph Ground["📚 2. Pre-Approved CIO House-View Grounding"]
        CVX --> GEN["Client Brief Narrative Engine\nExplains Verified Trades + Macro View"]
        CIO["Active CIO House-View Blocks\n(Compliance-Signed with TTL)"] --> GEN
    end

    subgraph Guard["⚖️ 3. FINRA 2210 Linter & SEC 17a-4 WORM Vault"]
        GEN --> LINT["FINRA 2210 / MiFID II Linter\nBlocks Promissory Terms + Adds Disclosures"]
        LINT --> WORM["SEC Rule 17a-4 WORM Archive\nLogs (IPS + Trades + CIO ID + Note)"]
        WORM --> FA["Salesforce FSC Advisor Workbench\n(3.5m Prep -> 1-Click Send)"]
    end
```

### 3.3 End-to-End Quarterly Review, Tax-Loss Harvesting, CIO Grounding & FINRA 2210 Sequence

```mermaid
sequenceDiagram
    autonumber
    participant FA as Financial Advisor (FA)
    participant Solver as IPS/APL + Tax-Lot Convex Solver
    participant CIO as CIO House-View & Narrative Engine
    participant Guard as FINRA 2210 Gate & WORM Vault
    participant Client as HNW Client Portal / Email

    FA->>Solver: Requests Q4 Review for `Household #4412 (Conservative R2, No Tobacco, <=5% Issuer Cap)`
    Solver->>Solver: Detects `+4%` tech overweight; solves tax-loss harvest within `APL` -> Swap into `AAA Tax-Exempt Muni ETF (MUB)`
    Solver->>CIO: Emits 100% `IPS`-compliant trade list + `+$4,200` tax-alpha calculation
    CIO->>CIO: Grounds narrative in active `CIO Block #2026-Q4-FI-08` ("Locking in tax-equivalent municipal yields")
    CIO->>Guard: Submits draft client portfolio review
    Guard->>Guard: `<45ms` verifies `0` promissory violations + appends Muni Interest-Rate Risk Disclosure + writes `SEC 17a-4 WORM` record
    Guard->>FA: Delivers approve-ready client brief in `3.5 minutes` (bypasses 2.8-day manual Series 24 queue)
    FA->>Client: Sends personalized review & 1-click trade authorization link
```

---

## 4. 📊 4-Tier Architecture Decision Matrix

| Evaluation Dimension | Tier 1: Unconstrained LLM Ticker Advice & Email | Tier 2: Legacy Manual Prep + `Series 24` Queue | Tier 3: CIO Research PDF RAG Only (No `IPS` Solver) | Tier 4: `IPS/APL` Solver + `CIO House-View` + `FINRA 2210` Gate (Recommended) |
| :--- | :--- | :--- | :--- | :--- |
| **`Reg BI / MiFID II` Suitability Breach** | **`16.8%` (off-APL / risk breach)** | `0.4%` | `8.2%` (misses concentration/tax caps) | **`0.0%` (100% mathematically bounded by `IPS + APL`)** |
| **`FINRA 2210` Promissory Breach Rate** | **`11.4%` (`"guaranteed"` / unbalanced)** | `0.6%` | `2.8%` | **`0.0%` (pre-approved CIO blocks + `<45ms` 2210 linter)** |
| **Advisor Meeting Prep Time** | `12.0 min` (heavy manual fixes) | **`52.0 min` (+ `2.8-day` compliance queue)** | `9.5 min` | **`3.5 min` (`15x` faster; pre-cleared blocks)** |
| **Tax-Lot & `Wash-Sale` Optimization** | ❌ Blind to tax lots | ⚠️ Manual spreadsheet errors | ❌ RAG cannot optimize tax lots | **✅ Deterministic `30-day Wash-Sale` & tax-loss harvesting** |
| **`SEC Rule 17a-4` Audit Reconstruction** | ❌ Unreproducible | ✅ Email archive | ⚠️ Document citations only | **✅ Immutable `(IPS_Snapshot, Trades, CIO_ID)` WORM record** |
| **Monthly Compliance & Advisor Prep TCO** | `$720,000 / mo` | `$540,000 / mo` | `$310,000 / mo` | **`$78,000 / mo` (`-89.2%` savings)** |

---

## 5. 🧪 Quantitative Case Study, Interactive Sizer & 7-Card Report

### 5.1 Baseline Workload (`12,000` Financial Advisors / `480,000` Client Households / `360,000` Monthly Reviews)
* **Workload Profile**: Global private bank and wealth management platform supporting `12,000` Financial Advisors (`$850B` AUM, `360,000` monthly client portfolio reviews and briefs across a `2,400`-security `APL`).
* **Reproducible Model & Dataset**: See [`experiments/enterprise_wealth_management_advisor_copilot_suitability_ips_guardrail_selection/run.py`](../../experiments/enterprise_wealth_management_advisor_copilot_suitability_ips_guardrail_selection/run.py) and [`results.json`](../../experiments/enterprise_wealth_management_advisor_copilot_suitability_ips_guardrail_selection/results.json) (`provenance: synthetic + anonymized_reported`, `independently_reproduced=false`).
* **Interactive Sizer**: 📊 [Open Enterprise Wealth Management Advisor Copilot Sizer (`enterprise-wealth-management-advisor-copilot-suitability-ips-guardrail-sizer.html`)](../reports/enterprise-wealth-management-advisor-copilot-suitability-ips-guardrail-sizer.html)
* **7-Card 4:5 Benchmark Deck**: 🃏 [Open 7-Card Visual Report (`enterprise-wealth-management-advisor-copilot-suitability-ips-guardrail-selection-cards.html`)](../reports/enterprise-wealth-management-advisor-copilot-suitability-ips-guardrail-selection-cards.html)

```chart
{
  "data": {
    "values": [
      { "tier": "Tier 1: Unconstrained LLM Advice", "suitability_breach_pct": 16.8, "promissory_breach_pct": 11.4, "prep_min": 12.0, "monthly_tco_k": 720.0 },
      { "tier": "Tier 2: Manual Prep + Series 24 Queue", "suitability_breach_pct": 0.4, "promissory_breach_pct": 0.6, "prep_min": 52.0, "monthly_tco_k": 540.0 },
      { "tier": "Tier 3: CIO PDF RAG Only", "suitability_breach_pct": 8.2, "promissory_breach_pct": 2.8, "prep_min": 9.5, "monthly_tco_k": 310.0 },
      { "tier": "Tier 4: IPS/APL Solver + CIO View + 2210 Gate", "suitability_breach_pct": 0.0, "promissory_breach_pct": 0.0, "prep_min": 3.5, "monthly_tco_k": 78.0 }
    ]
  },
  "semantic_types": {
    "tier": "Category",
    "suitability_breach_pct": "Percentage",
    "promissory_breach_pct": "Percentage",
    "prep_min": "Quantity",
    "monthly_tco_k": "Amount"
  },
  "chart_spec": {
    "chartType": "Bar Chart",
    "title": "Tier 4 Achieves 0.0% Reg BI / FINRA 2210 Violations, 3.5m Prep Time, and Cuts Monthly TCO by 89.2%",
    "subtitle": "Monthly compliance remediation, supervisory review, and advisor preparation TCO ($ USD Thousands)",
    "encodings": {
      "x": "tier",
      "y": "monthly_tco_k",
      "color": "tier"
    }
  }
}
```

---

## 6. 🛠️ Architecture Review & Production Readiness Checklist

- [ ] **Strict Separation of Portfolio Math (`IPS + APL` Solver) from Narrative Generation**: Is the LLM prohibited from selecting tickers or weights, with 100% of trade proposals computed deterministically inside the `Approved Product List (APL)` and client `IPS` bounds?
- [ ] **Versioned & TTL-Gated `CIO House-View` Narrative Repository**: Are expired Chief Investment Officer market views automatically evicted via metadata filters (`status == ACTIVE and expires_at > now()`)?
- [ ] **Pre-Send `<45ms` `FINRA Rule 2210 / MiFID II` Promissory-Language Gate**: Does a deterministic + NLI guardrail block promissory/exaggerated phrases (`"guaranteed"`, `"risk-free"`, `"assured return"`) and verify net-of-fees performance presentation?
- [ ] **Deterministic Asset-Class Risk Disclosure Injection**: Does code automatically append mandatory regulatory disclosures based on the exact asset classes referenced in the brief?
- [ ] **`SEC Rule 17a-4` Immutable `WORM` Provenance Logging**: Is every generated client communication archived alongside the exact `(Client_IPS_Snapshot, Tax_Lot_Trades, CIO_Block_IDs)` in tamper-evident `WORM` storage?
