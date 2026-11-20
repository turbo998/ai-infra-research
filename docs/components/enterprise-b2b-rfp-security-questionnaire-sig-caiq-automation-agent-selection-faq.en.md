# Enterprise B2B RFP, Tender & Security Questionnaire (`SIG / CAIQ / SOC2`) Automation Agent, Canonical Approved Answer Bank, `GA vs. Roadmap` Over-Commitment Firewall & Multi-Tab Excel Writeback Selection Guide

> **Language / 语言**: **English (Current)** | [中文](./enterprise-b2b-rfp-security-questionnaire-sig-caiq-automation-agent-selection-faq.md)  
> **Scope**: End-to-end production reference architecture and sizing playbook for Cloud & AI Solution Architects, Presales / Solutions Engineering Directors, and CISO / Trust-Center Teams automating B2B RFP, RFI, and Security Questionnaire responses (`800+` questionnaires / `200,000+` questions per month).  
> **Provenance**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`). Deterministic evaluation model sealed in [`experiments/enterprise_b2b_rfp_security_questionnaire_sig_caiq_automation_agent_selection/results.json`](../../experiments/enterprise_b2b_rfp_security_questionnaire_sig_caiq_automation_agent_selection/results.json).

![Enterprise B2B RFP & Security Questionnaire Automation Summary](../../assets/svg/enterprise-b2b-rfp-security-questionnaire-sig-caiq-automation-agent-selection-faq-summary.svg)

---

## ① 📌 Executive Summary (3-Minute Architect Verdict)

Every enterprise B2B deal requires completing detailed **Technical RFPs and Security Questionnaires (`SIG`, `CAIQ`, `HECVAT`, `SOC2 / ISO 27001` — `800` questionnaires/month averaging `250` questions = `200,000` monthly questions)** delivered inside locked multi-tab `.xlsx` workbooks or procurement portals. Three common AI automation approaches fail in production:
1. **Trap #1: Naive RAG Over Uncurated Internal Engineering Wikis, Slack & Past Proposals (`Tier 1`, `21.5%` Binding Over-Commitment Error Rate — Promising Unreleased `2027 Roadmap` Ideas or One-Off Custom Deal Concessions as Standard `GA` Product Features!, `$240,000/month` Legal Liability & Presales Rework TCO)**:
   - **The #1 Legal Liability in AI RFP Bots**: When an LLM searches raw Confluence wikis or old custom proposals, it finds a PM's roadmap brainstorm (*"Targeting FIPS 140-3 Level 3 in late 2027"*) or a custom SOW built for one client in 2023 and writes **`"Yes, supported in production"`** into a legally binding RFP response!
2. **Trap #2: Legacy Keyword Q&A Spreadsheet Search + Manual SE Copy-Paste (`Tier 2`, `228-Hour` / `9.5-Day` Turnaround Time, `52.0%` Escalated to Busy SEs/SMEs on Compound Multi-Part Questions, `$195,000/month` TCO)**.
3. **Trap #3: Uncalibrated `100%` Forced Auto-Fill Without Confidence Bands or Excel Dropdown Constraints (`Tier 3`, `12.8%` Over-Commitment Error, Breaks Customer Excel `DataValidation` Dropdowns, Forces SEs to Re-Read All 250 Rows, `$165,000/month` TCO)**.

Evaluated across **800 monthly RFPs and Security Questionnaires (`200,000` questions/month)**, the four architectures compare as follows:

| RFP & Questionnaire Tier | Knowledge Truth, Confidence Routing & Excel Stack | Over-Commitment Liability | Verified Auto-Completion | Turnaround Time | 800-RFP Monthly TCO | Production Verdict |
|---|---|---|---|---|---|---|---|
| **🟥 Tier 1: Naive RAG Over Uncurated Wikis & Old Proposals** | Searches raw Confluence/Slack & custom SOWs | **`21.5%` ✕** *(promises roadmap as GA!)* | `62.0%` | `72.0 hrs` *(manual audit)* | `$240,000 / mo` ✕ *(breach of contract)* | ⛔ Never index unreleased roadmap wikis or 1-off SOWs into RFP RAG |
| **🟨 Tier 2: Legacy Keyword Q&A Search + Manual Paste** | Keyword search on static bank; manual SE rewrite | `3.5%` | `48.0%` ✕ *(52% manual)* | `228.0 hrs` ✕ *(9.5 days!)* | `$195,000 / mo` ✕ *(burns SE bandwidth)* | ⛔ 9.5-day turnaround delays deal closing |
| **🟦 Tier 3: Forced `100%` Auto-Fill (`No Confidence Bands`)** | Fills every cell blindly; ignores `.xlsx` dropdowns | `12.8%` ✕ | `71.0%` | `48.0 hrs` | `$165,000 / mo` ✕ *(SE must re-check all)* | ⛔ Mixing low-confidence guesses with verified facts destroys SE trust |
| **🟩 Tier 4: `90-Day` Canonical Q&A Bank + `GA-Only` Firewall + 3-Band Confidence Router + `openpyxl` Dropdown Writeback** | **Quarantines roadmap/custom docs; `>=0.88` auto-green, `0.60-0.88` yellow draft, `<0.60` 1-click Slack SME loop** | **`0.0%` 🏆** *(zero roadmap leaks)* | **`91.5%` 🏆** *(compound sub-q synthesis)* | **`0.63 hrs (38m)` 🏆** *(360x faster!)* | **`$28,000 / mo` 🏆** *(-88.3% vs Tier 1)* | 🌟 **Recommended Enterprise RFP & Security Questionnaire Standard** |

> 💡 **Golden Rules for Enterprise B2B RFP & Security Questionnaire Automation**:
> 1. **Enforce a Hard `GA_STANDARD` vs. `ROADMAP / CUSTOM_SOW` Firewall**: **"Every 'Yes' in an RFP becomes a legally binding contract schedule!"** Restrict retrieval strictly to:
>    - **Tier-A (`Canonical Approved Q&A Bank`, `78%` hit rate)**: `2,500` CISO/Legal-attested question-answer pairs scoped by `product_line` and `deployment_mode (Cloud_SaaS vs. On_Prem)` with a mandatory **`90-day` freshness attestation TTL**.
>    - **Tier-B (`Official GA Documentation & SOC2 Type II Report`, `13.5%` hit rate)**: Filtered by `WHERE release_status == 'GA_STANDARD'`. Unreleased roadmap wikis and one-off custom SOWs are **physically excluded (`0.0%` binding over-commitment liability)**!
> 2. **3-Band Confidence Traffic-Light Routing + Self-Improving SME Flywheel**:
>    - **`🟢 Green Band (>= 0.88 confidence, 78% of questions)`**: Auto-filled and pre-approved with attached evidence citations (`SOC2_2026.pdf#p42`).
>    - **`🟡 Yellow Band (0.60–0.88 confidence, 13.5% compound questions)`**: Decomposed into sub-questions, synthesized from verified GA facts, and highlighted yellow for a 5-second SE review.
>    - **`🔴 Red Band (< 0.60 confidence, 8.5% novel edge cases)`**: Never guess! Leave blank and dispatch a **1-Click Slack/Teams Card** to the domain SME; when answered, auto-ingest the verified pair into the Canonical Bank!
> 3. **`openpyxl` Data-Validation Dropdown Enforcement + Automated Evidence Zip Bundler**: Inspect `ws.data_validations` on every target answer cell so `XGrammar` emits the exact dropdown enum required by the customer's spreadsheet (`"Yes" / "Partial" / "N/A"`), and auto-bundle all referenced `SOC2 / ISO / PenTest` PDFs in **`38 minutes`**!

---

## ② 🎯 Customer & Architect Deep-Dive Q&A (3 Preventing Cloud-vs-OnPrem Mixups, Decomposing Multi-Part Questions & Preserving Excel Dropdowns Pain Points)

### Q1: How do we prevent the agent from promising a Cloud-SaaS-only security feature when answering an RFP for our On-Premises / Self-Hosted product edition?
- **Questionnaire-Level Deployment & SKU Bitmap Scoping**:
  - Before processing row 1, the RFP workspace locks the target scope (`product_edition = "On_Prem_Self_Hosted"`, `region = "EU_Sovereign"`).
  - The vector and BM25 pre-filter enforces `WHERE 'On_Prem_Self_Hosted' IN applicable_editions`—ensuring cloud-only features (`"Managed Multi-Region Active-Active"`) are never retrieved for an on-premises bid!

### Q2: Why do legacy keyword Q&A banks (`Tier 2`) fail on `52%` of customer cells—and how does Sub-Question Decomposition (`Tier 4`) reach `91.5%` completion?
- **Decomposing Compound Multi-Part Cells into Atomic Canonical Lookups**:
  - Customers frequently pack 3 questions into a single Excel cell (*"Describe your encryption at rest, key rotation frequency, and BYOK HSM support"*). Keyword search fails because those 3 facts live in 3 separate canonical Q&A cards (`#102`, `#108`, `#115`).
  - Tier 4 decomposes the cell into `[q1, q2, q3]`, retrieves the 3 attested cards (`score >= 0.90` each), and synthesizes a structured 3-bullet response strictly grounded in those 3 cards!

### Q3: Why do automated Excel writeback scripts (`Tier 3`) frequently corrupt customer procurement workbooks?
- **Respecting `openpyxl` `DataValidation` Lists & Locked Formula Cells**:
  - Customer spreadsheets use hidden `COUNTIF(C10:C300, "Compliant")` scoring formulas that break if an LLM writes `"Yes - Fully Compliant"` into a cell whose `DataValidation` dropdown only allows `"Compliant"`. Tier 4 reads the cell's validation list and constrains the dropdown column via `XGrammar`, placing narrative explanations in the adjacent `Comments / Justification` column!

---

## ③ 🧭 Visual Architecture Diagrams (Decision Tree + Canonical Bank, `GA` Firewall & 3-Band Router Topology + Sequence)

### 3.1 Enterprise B2B RFP & Security Questionnaire Decision Tree

```mermaid
flowchart TD
    A["Enterprise B2B RFP & Security Questionnaire Selection<br/>(800 RFPs/Mo | 200,000 Questions/Mo | SIG / CAIQ / SOC2 & Multi-Tab Excel)"] --> B{"How are customer questions matched to product truth and written back?"}
    B -- "Naive RAG over uncurated engineering wikis & old proposals (Tier 1)" --> C["⛔ 21.5% Binding Over-Commitment Liability!<br/>Promises 2027 roadmap ideas & 1-off deals as GA ($240K/mo TCO)"]
    B -- "Legacy keyword Q&A search + manual SE copy-paste (Tier 2)" --> D["⛔ 228-Hour (9.5-Day) Turnaround & 52% Manual Escalations!<br/>Fails on compound multi-part questions ($195K/mo)"]
    B -- "Forced 100% auto-fill without confidence bands or dropdown checks (Tier 3)" --> E["⛔ 12.8% Over-Commitment & Broken Excel Dropdowns!<br/>Forces SEs to re-verify all 250 rows from scratch ($165K/mo)"]
    B -- "90d Canonical Q&A Bank + GA Firewall + 3-Band Router + Excel Writeback (Tier 4)" --> F["🟩 Tier 4: Canonical Q&A Bank + GA Firewall + 3-Band Router + Excel Writeback 🏆<br/>(0.0% Over-Commitment | 91.5% Completion | 38m Turnaround | -88.3% TCO)"]
    style F fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#f8fafc
    style C fill:#7f1d1d,stroke:#ef4444,color:#fecaca
    style D fill:#7f1d1d,stroke:#ef4444,color:#fecaca
    style E fill:#7f1d1d,stroke:#ef4444,color:#fecaca
```

### 3.2 Tier 4 `90-Day` Canonical Q&A Bank, `GA-Only` Firewall, 3-Band Confidence Router & `openpyxl` Writeback Architecture

```mermaid
flowchart LR
    subgraph Parse["1. Workbook Parser & Scope Lock"]
        XLS["📊 Customer RFP / SIG (.xlsx)<br/>250 Questions + Dropdowns"] --> Lock["🔒 Scope Bitmap & Validation Parser<br/>• Edition: SaaS vs. On-Prem<br/>• Extracts openpyxl dropdown enums"]
    end

    subgraph Truth["2. Two-Tier GA-Only Knowledge Hierarchy (0% Roadmap Leaks!)"]
        Lock --> Split["✂️ Compound Question Decomposer"]
        Split --> Bank[("🏆 Tier-A: 90d Attested Canonical Q&A Bank<br/>2,500 CISO/Legal-approved pairs + SOC2 refs")]
        Bank -- "Fallback if <0.88" --> GA[("📘 Tier-B: Official GA Product Docs<br/>WHERE status == 'GA_STANDARD'<br/>(Blocks Roadmap & 1-off SOWs!)")]
    end

    subgraph Route["3. 3-Band Traffic-Light Router & SME Flywheel (38 Min)"]
        Bank & GA --> Band{"🚦 Confidence Band"}
        Band -- ">=0.88 (78% Green)" --> G["🟢 Auto-Approve + Attach SOC2 PDF"]
        Band -- "0.60–0.88 (13.5% Yellow)" --> Y["🟡 Highlight Synthesized Draft for SE"]
        Band -- "<0.60 (8.5% Red)" --> R["🔴 1-Click Slack SME Card -> Auto-Update Bank!"]
    end
```

### 3.3 End-to-End Sequence: Blocking a Roadmap Over-Commitment & Completing a 250-Question `SIG` Workbook in `38 Minutes`

```mermaid
sequenceDiagram
    autonumber
    participant SE as Solutions Engineer (Customer SIG_RFP.xlsx)
    participant Parser as openpyxl Validation & Question Parser
    participant Bank as Canonical Q&A Bank + GA Firewall
    participant SME as Security SME (Slack Card for 8.5% Red Band)
    participant Out as Excel Writeback & Evidence Bundler

    SE->>Parser: 1. Upload 250-question `Bank_Vendor_SIG.xlsx` scoped to `Edition = Cloud_SaaS_EU`
    Parser->>Bank: 2. Parse Row #88: Dropdown allows `["Yes", "No", "Compensating"]`; Question = `"Is FIPS 140-3 Level 3 live in EU?"`
    Bank->>Bank: 3. Enforce `WHERE status == 'GA_STANDARD'` -> block 2027 Roadmap Wiki (`0.0%` false `"Yes"` commitment!)
    Bank-->>Out: 4. Fill `78%` Green rows (`>=0.88`) with exact dropdown enum + justification + `SOC2_Type2.pdf#p42` ref; flag `13.5%` Yellow rows
    Bank->>SME: 5. Dispatch `8.5%` novel Red rows to Security SME in Slack -> SME answers in 12m & clicks `[Save to Canonical Bank]`
    Out-->>SE: 6. `T + 38m`: Export formula-intact `Bank_Vendor_SIG_Completed.xlsx` + `Evidence_Package.zip` (`91.5%` auto-completion)!
```

---

## ④ 📊 4-Tier Architecture Decision Matrix

| Evaluation Dimension | 🟥 Tier 1: Uncurated Wiki RAG | 🟨 Tier 2: Keyword DB + Manual | 🟦 Tier 3: Forced 100% Auto-Fill | 🟩 Tier 4: Canonical Bank + `GA` Firewall + 3-Band |
|---|---|---|---|---|
| **Binding Over-Commitment Rate** | `21.5%` ✕ (promises roadmap as GA) | `3.5%` | `12.8%` ✕ (hallucinates edge cases) | 🏆 **`0.0%` (quarantines roadmap/custom SOWs; GA truth only)** |
| **Verified Auto-Completion Rate** | `62.0%` | `48.0%` ✕ (fails compound cells) | `71.0%` | 🏆 **`91.5%` (sub-question decomposition + SME flywheel)** |
| **Excel Dropdown & Formula Safety** | Breaks validation rules | Manual | Corrupts dropdowns | 🏆 **`100%` (`openpyxl` `DataValidation` enum constraint)** |
| **250-Question RFP Turnaround** | `72.0 hrs` (manual audit) | `228.0 hrs` ✕ (9.5 days) | `48.0 hrs` | 🏆 **`0.63 hrs (38 minutes, 360x faster than Tier 2)`** |
| **800-RFP Monthly TCO** | `$240,000 / mo` ✕ | `$195,000 / mo` ✕ | `$165,000 / mo` ✕ | **`$28,000 / mo` 🏆 (-88.3% vs Tier 1)** |

---

## ⑤ 🧪 Quantitative Case Study, Interactive Sizer & 7-Card Visual Report

- **Interactive HTML Sizer**: 👉 [Open Enterprise B2B RFP & Security Questionnaire Agent Sizer](../reports/enterprise-b2b-rfp-security-questionnaire-sig-caiq-automation-agent-sizer.html)
- **7-Card 4:5 Visual Benchmark Report**: 👉 [Open 7-Card Visual Benchmark Deck](../reports/enterprise-b2b-rfp-security-questionnaire-sig-caiq-automation-agent-selection-cards.html)
- **Deterministic Verification**:
  ```bash
  python experiments/enterprise_b2b_rfp_security_questionnaire_sig_caiq_automation_agent_selection/run.py --check
  python -m unittest tests/test_enterprise_b2b_rfp_security_questionnaire_sig_caiq_automation_agent_selection.py
  ```

---

## ⑥ 🛠️ Production Readiness & Architecture Review Checklist

- [ ] **Physically Quarantine `ROADMAP`, `BETA`, and `CUSTOM_SOW` Documents from `GA_STANDARD` Truth**: Never allow an RFP agent to cite internal engineering wikis or one-off historical concessions as standard product capabilities (`0.0%` over-commitment liability).
- [ ] **Maintain a `90-Day` SME-Attested Canonical Q&A Bank Scoped by `Product Edition` and `Region`**: Automatically demote expired Q&A pairs to the Yellow review band after 90 days.
- [ ] **Decompose Compound Multi-Part Cells Before Retrieval**: Split 3-in-1 customer questions into atomic lookups against the Canonical Bank (`91.5%` verified completion).
- [ ] **Route Responses via a 3-Band Confidence Traffic Light (`>=0.88` Auto-Green / `0.60–0.88` Yellow Review / `<0.60` Red Slack SME Escalation)**: Auto-ingest new SME answers from Slack directly back into the Canonical Bank.
- [ ] **Enforce Customer `.xlsx` `DataValidation` Dropdown Enums and Auto-Bundle `SOC2 / ISO / PenTest` Evidence Exhibits**: Deliver upload-ready spreadsheets and proof packages in `38 minutes`.
