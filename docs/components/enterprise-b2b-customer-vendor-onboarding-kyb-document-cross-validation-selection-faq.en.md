# Enterprise B2B Customer & Vendor Onboarding (`KYB` Multi-Document Cross-Validation), Ultimate Beneficial Owner (`UBO >= 25%`) Graph Traversal & Automated Deficiency-Chase Agent Architecture Selection Guide

> **Language / 语言**: **English (Current)** | [中文](./enterprise-b2b-customer-vendor-onboarding-kyb-document-cross-validation-selection-faq.md)  
> **Scope**: End-to-end production reference architecture and sizing playbook for Cloud & AI Solution Architects, FinTech/Banking `KYB & AML` Risk Leads, and Enterprise Procurement (`SAP Ariba / Coupa` Vendor Master) Engineers automating multi-document corporate onboarding (`30,000+` monthly packets).  
> **Provenance**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`). Deterministic evaluation model sealed in [`experiments/enterprise_b2b_customer_vendor_onboarding_kyb_document_cross_validation_selection/results.json`](../../experiments/enterprise_b2b_customer_vendor_onboarding_kyb_document_cross_validation_selection/results.json).

![Enterprise B2B Customer & Vendor Onboarding KYB Summary](../../assets/svg/enterprise-b2b-customer-vendor-onboarding-kyb-document-cross-validation-selection-faq-summary.svg)

---

## ① 📌 Executive Summary (3-Minute Architect Verdict)

In commercial banking, B2B payment merchant onboarding (`KYB`), and enterprise supplier master registration (**`30,000` monthly onboarding packets, each containing `6` heterogeneous documents / `48` pages: `Certificate of Incorporation`, `Share Register / Operating Agreement`, `W-9 / W-8BEN-E Tax Form`, `Voided Check / Bank Letter`, `Director Passport/ID`, and `SOC2 / ISO Certificate`**), three common architectures fail in production:
1. **Trap #1: Manual Back-Office Document Review & Multi-Day Email Deficiency Ping-Pong (`Tier 1`, `204 Hours` = `8.5 Days` Average Turnaround Time, `42.0%` Deficiency Re-Submission Rate, `$265,000/month` Ops & Customer Drop-Off TCO)**:
   - Customers upload a blurry bank letter or an unsigned W-9 on Friday and wait 3 business days just to receive an email asking them to re-upload!
2. **Trap #2: Isolated Per-Document OCR/LLM Extraction + LLM Mental Math for Multi-Layer `UBO` Equity (`Tier 2`, Misses `14.8%` of Cross-Document Bank-Account Substitution Fraud AND Fails `28.5%` of Multi-Layer `UBO >= 25%` Calculations, `$172,000/month` TCO)**:
   - **The #1 B2B Onboarding Fraud Vector**: Submitting genuine incorporation and tax documents for `"Acme Logistics Inc"` alongside a bank routing letter belonging to a fraudster's shell company `"Acme Logistics Holdings LLC"`. Isolated per-document validators mark all 6 documents as `"Valid"`!
   - Furthermore, asking an LLM to mentally multiply and sum indirect ownership across 4-layer diamond or cyclic holding companies (`FinCEN / EU AMLD 25% UBO rule`) produces a **`28.5%` calculation error rate**!
3. **Trap #3: Stuffing All 80 Pages of 6–10 PDFs into a Single Monolithic Vision LLM Prompt (`Tier 3`, `$198,000/month` VLM Token Bill, Still Suffers `19.4%` UBO Math Error)**.

Evaluated across **30,000 monthly B2B onboarding packets (`6` documents/packet, `25%` regulatory UBO threshold)**, the four architectures compare as follows:

| Onboarding Tier | Pre-Flight, Cross-Doc Validation & UBO Stack | Cross-Doc Fraud Catch | Multi-Layer `UBO` Math Error | Turnaround Time | 30K-Packet Monthly TCO | Production Verdict |
|---|---|---|---|---|---|---|---|
| **🟥 Tier 1: Manual Back-Office Review & Email Chase** | Human analysts read 6 PDFs & email for missing signatures | `93.6%` *(human fatigue)* | `8.5%` *(complex holding math)* | `204.0 hrs` ✕ *(8.5 days!)* | `$265,000 / mo` ✕ *(ops + customer churn)* | ⛔ 8.5-day onboarding bottleneck kills B2B conversion |
| **🟨 Tier 2: Isolated Per-Document LLM Extraction** | Validates each doc in isolation; LLM mental math for UBO | `85.2%` ✕ *(misses 14.8% bank swap)* | **`28.5%` ✕** *(fails diamond graphs)* | `18.0 hrs` | `$172,000 / mo` ✕ *(AML & fraud liability)* | ⛔ Blind to cross-document entity swaps and multi-hop equity math |
| **🟦 Tier 3: Monolithic 80-Page Multi-Doc `VLM` Stuffing** | Dumps all 48–80 pages as images into one VLM prompt | `91.0%` | `19.4%` ✕ | `6.5 hrs` | `$198,000 / mo` ✕ *(massive VLM bill)* | ⛔ Expensive vision tokens still cannot do exact matrix algebra |
| **🟩 Tier 4: `1.8s` Upload Pre-Flight + 6-Way Entity Matrix + Live Registry API + `NetworkX` Matrix `UBO` Solver** | **In-browser `<2s` quality gate + deterministic 6-doc entity reconciliation + $(I-A)^{-1}-I$ equity solver** | **`99.7%` 🏆** *(catches shell swaps)* | **`0.0%` 🏆** *(exact linear algebra)* | **`0.23 hrs (14m)` 🏆** *(886x faster!)* | **`$32,000 / mo` 🏆** *(-87.9% vs Tier 1)* | 🌟 **Recommended Enterprise B2B KYB & Vendor Onboarding Standard** |

> 💡 **Golden Rules for Enterprise B2B Customer & Vendor Onboarding (`KYB`)**:
> 1. **`1.8-Second` In-Browser Upload Pre-Flight Gate**: **"Catch blurry scans and missing signatures before the user closes the browser tab!"** Run a fast `<2s` pre-flight check (`DPI blur detection + document-type classifier + signature/stamp detector + expiry date check`) on upload—slashing multi-day email deficiency ping-pong from `42.0%` to **`3.5%`**!
> 2. **Deterministic 6-Way Cross-Document Entity Reconciliation Matrix + Live Registry API**: Reconcile `Legal Entity Name`, `Tax ID (EIN/VAT)`, `Registered Address`, `Authorized Signatory`, and `Bank Account Beneficiary` across all 6 documents AND verify against live government/banking registries (`OpenCorporates / Secretary of State / IRS TIN / EU VIES / Early Warning Account Owner`)—catching **`99.7%` of cross-document shell-company substitution fraud**!
> 3. **LLM Extracts Ownership Edges; `NumPy / NetworkX` Matrix Power Series Solves `UBO >= 25%` ($W = (I - A)^{-1} - I$)**: Never let an LLM multiply percentages across holding trees! Have the extractor emit directed edges `(owner_i -> entity_j, share_pct)` and solve total direct + indirect ownership via matrix inversion $(I - A)^{-1} - I$ in `0.5ms` with **`0.0%` arithmetic error**!

---

## ② 🎯 Customer & Architect Deep-Dive Q&A (3 Shell-Company Bank Substitution, Diamond/Cyclic `UBO` Graphs & `14-Minute` Straight-Through Processing Pain Points)

### Q1: How does a fraudster pass isolated document checks (`Tier 2`), and how does the 6-Way Cross-Document Matrix (`Tier 4`) catch `99.7%` of these attacks?
- **Defeating Cross-Document Identity & Bank Substitution**:
  - In Tier 2, each document is validated by an independent prompt: the `W-9` is valid for `"Acme Logistics Inc (EIN 12-3456789)"`, and the `Voided Check` is valid for `"Acme Logistics LLC"`—so both pass!
  - Tier 4 populates a **Canonical Entity Matrix** across all 6 documents and enforces strict cross-document invariants: `canonical_legal_name(W9) == canonical_legal_name(Incorporation) == canonical_legal_name(Bank_Letter)` (`"Inc"` != `"LLC"`, flagging an immediate high-severity mismatch!) plus live `IRS TIN / VIES` registry verification!

### Q2: Why do LLMs fail `28.5%` of multi-layer Ultimate Beneficial Owner (`UBO >= 25%`) calculations—and how does $(I - A)^{-1} - I$ achieve `0.0%` error?
- **Diamond Ownership Paths & Circular Cross-Holdings**:
  - Suppose Individual `P1` owns `12%` of `Applicant` directly, and also owns `70%` of `HoldCo_A`, which owns `50%` of `HoldCo_B`, which owns `40%` of `Applicant` ($0.70 \times 0.50 \times 0.40 = 14\%$). LLMs frequently inspect each path separately (`12% < 25%` and `14% < 25%`) and falsely conclude that `P1` is below the `25%` FinCEN UBO threshold (missing that $12\% + 14\% = \mathbf{26.0\% \ge 25\%}$)!
  - In Tier 4, the LLM only extracts the pairwise adjacency table $A_{ij}$. A deterministic `NumPy / NetworkX` solver computes the Neumann series $W_{\text{total}} = \sum_{k=1}^\infty A^k = (I - A)^{-1} - I$, handling arbitrary multi-hop diamond and circular ownership graphs with **`0.0%` math error**!

### Q3: How do we compress B2B onboarding turnaround from `8.5 days` (`204 hours`) down to `14 minutes` (`0.23 hours`)?
- **Eliminating the `42%` Deficiency Email Loop + `82%` Straight-Through Processing (`STP`)**:
  - By catching unsigned forms and low-DPI scans in `1.8 seconds` during upload, `96.5%` of submitted packets enter the parallel extraction + registry + `OFAC` screening pipeline cleanly. Clean packets (`82%` of volume) pass all deterministic matrix gates in **`14 minutes`**, while only the `18%` with genuine ownership/sanctions flags are routed to compliance officers with a pre-rendered interactive UBO graph!

---

## ③ 🧭 Visual Architecture Diagrams (Decision Tree + `1.8s` Pre-Flight, 6-Way Matrix & `UBO` Solver Topology + Sequence)

### 3.1 Enterprise B2B Customer & Vendor Onboarding (`KYB`) Decision Tree

```mermaid
flowchart TD
    A["Enterprise B2B KYB & Vendor Onboarding Selection<br/>(30,000 Packets/Mo | 6 Docs / 48 Pages per Packet | 25% Regulatory UBO Rule)"] --> B{"How are multi-doc packets validated and multi-layer UBOs computed?"}
    B -- "Manual back-office review & email deficiency chase (Tier 1)" --> C["⛔ 204-Hour (8.5-Day) Turnaround & 42% Email Re-Work!<br/>Severe B2B customer drop-off ($265K/mo TCO)"]
    B -- "Isolated per-document LLM extraction + LLM UBO mental math (Tier 2)" --> D["⛔ Misses 14.8% Cross-Doc Bank Swaps + 28.5% UBO Math Error!<br/>Fails on shell-LLC bank substitution & diamond equity graphs ($172K/mo)"]
    B -- "Monolithic 80-page multi-doc VLM prompt stuffing (Tier 3)" --> E["⛔ $198K/mo VLM Token Bill & 19.4% UBO Math Error!<br/>Vision attention degrades across 80 pages"]
    B -- "1.8s Upload Pre-Flight + 6-Way Entity Matrix + Registry API + NetworkX UBO DAG (Tier 4)" --> F["🟩 Tier 4: 1.8s Pre-Flight + 6-Way Entity Matrix + Matrix UBO Solver 🏆<br/>(99.7% Fraud Catch | 0.0% UBO Math Error | 14-Min Turnaround | -87.9% TCO)"]
    style F fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#f8fafc
    style C fill:#7f1d1d,stroke:#ef4444,color:#fecaca
    style D fill:#7f1d1d,stroke:#ef4444,color:#fecaca
    style E fill:#7f1d1d,stroke:#ef4444,color:#fecaca
```

### 3.2 Tier 4 `1.8s` Upload Pre-Flight, 6-Way Entity Reconciliation Matrix & `NetworkX UBO` Graph Solver Architecture

```mermaid
flowchart LR
    subgraph Pre["1. 1.8s In-Browser Pre-Flight Gate"]
        Up["📤 Upload 6 KYB Documents<br/>(Incorp, Share Reg, W-9, Bank, ID, Cert)"] --> Check["⚡ 1.8s DPI / Signature / Expiry Gate<br/>Rejects blurry or unsigned PDFs live<br/>(Deficiency loop 42% -> 3.5%!)"]
    end

    subgraph Recon["2. Parallel Extraction & 6-Way Entity Matrix"]
        Check --> Par["🔍 6-Way Parallel Doc Extractor"]
        Par --> Mat["📐 Cross-Doc Entity Consistency Matrix<br/>Reconciles [Legal Name, TIN, Address, Bank Holder]<br/>+ Live Govt Registry / TIN API (99.7% catch!)"]
    end

    subgraph UBO["3. Exact Matrix UBO Solver & AML Screening"]
        Par --> Solver["🕸️ NetworkX / NumPy UBO Solver<br/>W = (I - A)^(-1) - I (0.0% math error!)<br/>Extracts all UBOs >= 25.0%"]
        Mat & Solver --> Screen["🛡️ Automated OFAC / EU / PEP Screen<br/>-> 82% Auto-Approve (14m) | 18% Review"]
    end
```

### 3.3 End-to-End Sequence: Live Signature Pre-Flight, `26.0%` Diamond `UBO` Graph Resolution & `14-Minute` Approval

```mermaid
sequenceDiagram
    autonumber
    participant Applicant as B2B Applicant (Portal UI)
    participant Gate as 1.8s Upload Pre-Flight Service
    participant Matrix as 6-Way Extraction & Entity Matrix
    participant UBO as NetworkX Matrix UBO Solver
    participant Reg as Live Govt Registry & OFAC API

    Applicant->>Gate: 1. Upload 6 onboarding PDFs (W-9 is missing handwritten signature)
    Gate-->>Applicant: 2. `1.8s` instant inline prompt: `"W-9 signature box is blank—please sign & re-upload now"` (`0` email delay!)
    Applicant->>Matrix: 3. Upload signed W-9 -> run parallel extraction across all 6 documents (`8s`)
    Matrix->>UBO: 4. Pass extracted shareholding edges -> compute $(I-A)^{-1}-I$ -> identify `Alice = 12% direct + 14% indirect = 26.0% (>= 25% UBO)`!
    UBO->>Reg: 5. Verify `Applicant EIN` + screen `Applicant` & `UBO (Alice)` against live Registry & OFAC Sanctions APIs
    Matrix->>Applicant: 6. Verify 6-way `Legal Name + TIN + Bank Beneficiary` match -> approve account in `14 minutes` (`99.7%` fraud catch)!
```

---

## ④ 📊 4-Tier Architecture Decision Matrix

| Evaluation Dimension | 🟥 Tier 1: Manual Review | 🟨 Tier 2: Isolated Per-Doc LLM | 🟦 Tier 3: 80-Page Giant `VLM` | 🟩 Tier 4: `1.8s` Pre-Flight + Matrix + `UBO` DAG |
|---|---|---|---|---|
| **Cross-Doc Fraud / Mismatch Catch** | `93.6%` (fatigue misses) | `85.2%` ✕ (misses bank swaps) | `91.0%` | 🏆 **`99.7%` (6-way entity matrix + live registry API)** |
| **Multi-Layer `UBO` Math Error** | `8.5%` | `28.5%` ✕ (fails diamond graphs) | `19.4%` ✕ | 🏆 **`0.0%` (`NumPy / NetworkX` $(I-A)^{-1}-I$ solver)** |
| **Post-Submission Deficiency Loop** | `42.0%` ✕ (multi-day emails) | `24.0%` | `18.0%` | 🏆 **`3.5%` (caught in `1.8s` during browser upload)** |
| **Onboarding Turnaround Time** | `204.0 hrs` ✕ (8.5 days) | `18.0 hrs` | `6.5 hrs` | 🏆 **`0.23 hrs (14 minutes, 886x faster than manual)`** |
| **30K-Packet Monthly TCO** | `$265,000 / mo` ✕ | `$172,000 / mo` ✕ | `$198,000 / mo` ✕ | **`$32,000 / mo` 🏆 (-87.9% vs Tier 1)** |

---

## ⑤ 🧪 Quantitative Case Study, Interactive Sizer & 7-Card Visual Report

- **Interactive HTML Sizer**: 👉 [Open Enterprise B2B KYB & Vendor Onboarding Sizer](../reports/enterprise-b2b-customer-vendor-onboarding-kyb-document-cross-validation-sizer.html)
- **7-Card 4:5 Visual Benchmark Report**: 👉 [Open 7-Card Visual Benchmark Deck](../reports/enterprise-b2b-customer-vendor-onboarding-kyb-document-cross-validation-selection-cards.html)
- **Deterministic Verification**:
  ```bash
  python experiments/enterprise_b2b_customer_vendor_onboarding_kyb_document_cross_validation_selection/run.py --check
  python -m unittest tests/test_enterprise_b2b_customer_vendor_onboarding_kyb_document_cross_validation_selection.py
  ```

---

## ⑥ 🛠️ Production Readiness & Architecture Review Checklist

- [ ] **Embed a `<2-Second` Upload Pre-Flight Gate (`DPI Blur + Doc Classifier + Signature/Stamp Detector + Expiry Check`) in the Portal UI**: Catch 90% of document quality defects before the applicant leaves the page.
- [ ] **Enforce a Deterministic 6-Way Cross-Document Entity Reconciliation Matrix**: Verify that `Legal Entity Name`, `Tax ID (EIN/VAT)`, `Registered Address`, and `Bank Account Beneficiary` match across all uploaded PDFs.
- [ ] **Compute Multi-Layer `UBO >= 25%` Ownership via Matrix Algebra ($W = (I - A)^{-1} - I$), Never LLM Mental Math**: Use the LLM strictly to extract `(parent -> child, share_pct)` edges and let `NumPy / NetworkX` solve indirect diamond and cyclic holdings (`0.0%` math error).
- [ ] **Cross-Verify Extracted Entities Against Live Authoritative Registries (`Secretary of State / OpenCorporates / IRS TIN / EU VIES / Bank Account Ownership`)**: Defeat altered PDFs and synthetic shell companies.
- [ ] **Screen Both the Applicant Entity and All Computed `>=25%` Natural-Person UBOs Against `OFAC / EU / UN / PEP` Lists Automatically**: Route clean packets (`82%`) through 14-minute Straight-Through Processing (`STP`) and surface visual UBO graphs for the `18%` requiring compliance review.
