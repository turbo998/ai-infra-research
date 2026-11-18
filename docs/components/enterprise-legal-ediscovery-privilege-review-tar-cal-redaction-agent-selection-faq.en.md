# Enterprise Legal E-Discovery, Litigation Privilege Review (`Attorney-Client / Work-Product`), `TAR 2.0` Continuous Active Learning (`CAL`) & Redaction Leak-Prevention Agent Architecture Selection Guide: Boolean Keywords + Linear Human Review vs. Standalone Zero-Shot LLM vs. Legacy `TAR 1.0` vs. `TAR 2.0 CAL` + Counsel-Graph & Family-Attachment Privilege Lock + `300-DPI` Raster Redaction Scrubber + `FRCP Rule 26(b)(5)` Privilege-Log Agent

> **Language / 语言**: **English (Current)** | [中文](./enterprise-legal-ediscovery-privilege-review-tar-cal-redaction-agent-selection-faq.md)  
> **Scope**: End-to-end production reference architecture and TCO selection guide for Cloud & AI Solution Architects, Corporate General Counsel & VPs of Litigation, Law Firm E-Discovery Practice Leads, and Legal Tech (`Relativity / Reveal / Everlaw`) Engineering Teams.  
> **Provenance**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`). Deterministic evaluation model sealed in [`experiments/enterprise_legal_ediscovery_privilege_review_tar_cal_redaction_agent_selection/results.json`](../../experiments/enterprise_legal_ediscovery_privilege_review_tar_cal_redaction_agent_selection/results.json).

![Enterprise Legal E-Discovery, Privilege Review, TAR 2.0 CAL & Redaction Agent Summary](../../assets/svg/enterprise-legal-ediscovery-privilege-review-tar-cal-redaction-agent-selection-faq-summary.svg)

---

## ① 📌 Executive Summary (3-Minute Architect Verdict)

When responding to a **DOJ/FTC Antitrust Second Request, patent litigation, or class-action subpoena** across **2.4 million collected custodian documents (`42,000` privileged documents, `$95/hr` contract attorney review rate, and a 6-week court production deadline)**, General Counsel and litigation support teams face three catastrophic failure modes:
1. **Negotiated Boolean Keyword Search Misses `42.0%` of Responsive Documents (`58.0%` Recall), While Linear Human Review Burns `$1.15M/Month`**: Contract attorneys reading linearly at 45 documents/hour suffer severe fatigue, missing `3.4%` of privileged documents.
2. **Standalone Zero-Shot LLM Prompts Cause `6.8%` Inadvertent Attorney-Client Privilege Waivers**: When a VP of Product emails a VP of Sales stating *"Per Sarah's guidance yesterday, pause the API launch until the patent indemnification clause is revised"* (where *Sarah* is External Patent Counsel, not on the `From/To/CC` header), a standalone document-by-document LLM (Tier 2) sees two business executives and classifies the email as `Non-Privileged`—producing it to opposing counsel and risking **subject-matter privilege waiver**!
3. **Vector PDF Overlay Redactions & Hidden Excel Pivot Caches Leak Secrets (`2.8%–4.1%` Leak Rate)**: Drawing black vector rectangles over PDF text without rasterizing leaves the underlying text stream intact—allowing opposing counsel or journalists to copy-paste "redacted" trade secrets and SSNs in seconds!

Evaluated across a **2.4-million-document custodian collection with 42,000 privileged documents**, the four architectural tiers compare as follows:

| Architectural Tier | Core Stack Combination | Responsive Document Recall | Privilege Waiver Escape Rate | Redaction Text-Layer Leak Rate | Human Review Volume Cut | Monthly Review & Sanction Risk TCO | Production Verdict |
|---|---|---|---|---|---|---|---|
| **🟥 Tier 1: Boolean Keywords + Linear Human Review** | Negotiated keyword strings + 120 contract attorneys reading linearly + manual PDF boxes | `58.0%` ✕ *(misses 42%)* | `3.4%` ✕ *(reviewer fatigue)* | `2.8%` ✕ *(copy-paste leak)* | `0.0%` *(full linear read)* | `$1,150,000 / mo` ✕ | ⛔ Low recall & runaway review fees |
| **🟨 Tier 2: Standalone Zero-Shot General LLM** | Prompting an LLM per document without counsel graphs, family locks, or elusion proof | `76.5%` | `6.8%` ✕ *(misses relayed advice)* | `4.1%` ✕ | `65.0%` | `$740,000 / mo` ✕ | ⛔ Waives privilege on non-lawyer relays |
| **🟦 Tier 3: Legacy `TAR 1.0` Static Seed Set** | Upfront static control set without CAL, attachment-family locks, or raster burn | `81.0%` | `2.1%` ✕ | `1.4%` ✕ | `58.0%` | `$495,000 / mo` ✕ | ⚠️ Produces privileged attachments in split threads |
| **🟩 Tier 4: `TAR 2.0 CAL` + Counsel-Graph Lock + `300-DPI` Raster Burn + `FRCP 26(b)(5)` Agent** | **`TAR 2.0 CAL` (Elusion Stopping Proof) + Counsel Entity Graph & Family/Near-Dup Lock + `300-DPI` Raster Scrubber + `FRCP 26(b)(5)` Logger** | **`96.4%` 🏆** *(+38.4 pts)* | **`0.0%` 🏆** *(Zero waiver)* | **`0.0%` 🏆** *(Bitmap burn)* | **`84.5%` 🏆** *(15.5% reviewed)* | **`$138,000 / mo` 🏆** *(-88.0%)* | 🌟 **Recommended Architecture** |

> 💡 **Golden Rule for Legal E-Discovery & Privilege Review Architects**: **"Never evaluate privilege on isolated documents without an Entity-Counsel Graph and Parent-Child Attachment Family closure lock, never stop review without a statistically defensible `TAR 2.0 CAL` binomial elusion test, and never produce a redacted document without burning redacted pages into `300-DPI` flattened bitmaps verified by post-burn OCR."**

---

## ② 🎯 Customer & Architect Deep-Dive Q&A (3 Real-World E-Discovery & Privilege Pain Points)

### Q1: How do the Entity-Counsel Directory Graph and Document-Family Closure Lock prevent the `6.8%` privilege waiver rate of standalone LLMs?
- **First-Principles Reality**:
  1. **Catching Implicit Relays of Legal Advice**: Non-lawyers frequently forward or summarize legal counsel's advice to other business executives without CC'ing the attorney (*"Per Sarah's guidance yesterday..."*). Tier 4 resolves first names, initials, and nicknames against the **Matter Entity-Counsel Directory Graph** and backward email-thread ancestry—flagging implicit legal-advice relays as `WITHHOLD_PRIVILEGED`.
  2. **Parent-Child Family & `95% MinHash` Near-Duplicate Closure Lock**: Before Bates stamping, Tier 4 enforces a deterministic boolean closure across every parent email, child attachment, and `95% MinHash` near-duplicate thread:
     $$\text{Priv\_Lock}(d) = \bigvee_{m \in \text{Family}(d) \cup \text{NearDup}_{95\%}(d)} \text{Is\_Privileged}(m)$$
     guaranteeing that a privileged Word attachment is never withheld in Thread A while accidentally produced in Thread B (`0.0%` waiver escape).

### Q2: How does `TAR 2.0 Continuous Active Learning (CAL)` with Binomial Elusion Sampling cut human review volume by `84.5%` while achieving `96.4%` court-defensible recall?
- **Continuous Re-Ranking + Statistical Stopping Proof**:
  - Unlike legacy `TAR 1.0` (which wastes thousands of hours on static random seed sets), `TAR 2.0 CAL` continuously retrains on every batch of 200 human-coded documents, surfacing high-probability responsive documents to the top of the review queue.
  - Once the rolling precision of newly surfaced documents drops below `2%` (after reviewing just **`15.5%`** of the 2.4M corpus), Tier 4 draws a **stratified random Elusion Sample ($N = 1,500$)** from the unreviewed null set, proving with 95% binomial confidence that overall recall is **`96.4%`**.

### Q3: How does the `300-DPI` Raster Burn Scrubber eliminate copy-paste PDF redaction leaks and hidden Excel metadata?
- **Destroying Underlying Vector Text Streams & `veryHidden` OOXML Sheets**:
  - Tier 4 unpacks native `OOXML` files to expose `veryHidden` Excel sheets, hidden pivot caches, speaker notes, and white-on-white text prior to review.
  - For every redacted document, Tier 4 **rasterizes the page to a flattened `300-DPI Group-4 TIFF / Image PDF` bitmap** (completely removing the vector PDF text layer) and runs a **post-burn OCR verification pass** over every redaction bounding box to assert `character_count == 0` (`0.0%` redaction leak).

---

## ③ 🧭 Visual Architecture Diagrams (Decision Tree + TAR 2.0/Privilege/Redaction Topology + Sequence Flow)

### 3.1 Architecture Selection Decision Tree

```mermaid
flowchart TD
    A["Enterprise Litigation E-Discovery & Privilege Review<br/>(2.4M Custodian Docs + 42,000 Privileged Docs + 6-Week Deadline)"] --> B{"Uses Boolean keyword filters + linear human review<br/>or standalone per-document zero-shot LLMs?"}
    B -- "Yes (Keyword filters or isolated LLM)" --> C["⛔ Low Recall & Privilege Waiver (Tier 1 / Tier 2)!<br/>58.0% keyword recall or 6.8% privilege waiver on relayed legal advice"]
    B -- "Uses predictive coding (TAR)" --> D{"Combines TAR 2.0 CAL elusion proof, Counsel-Graph family lock,<br/>300-DPI raster redaction burn & FRCP 26(b)(5) logger?"}
    D -- "No (Legacy TAR 1.0 + vector overlay redaction)" --> E["⚠️ Split-Thread Waivers & Copy-Paste Leaks (Tier 3)<br/>2.1% privilege escape & 1.4% PDF text-layer redaction leaks"]
    D -- "Yes (TAR 2.0 CAL + Counsel Graph + Raster Burn)" --> F["🟩 Tier 4: TAR 2.0 CAL + Counsel-Graph Family Lock + 300-DPI Raster Burn 🏆<br/>(96.4% Recall | -84.5% Review Volume | 0.0% Waiver/Redaction Leak)"]
    style F fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#f8fafc
    style C fill:#7f1d1d,stroke:#ef4444,color:#fecaca
```

### 3.2 Tier 4 `TAR 2.0 CAL` + Counsel-Graph Privilege Lock + `300-DPI` Raster Burn Topology

```mermaid
flowchart LR
    subgraph Ingest["1. Deep OOXML Unpack & TAR 2.0 CAL"]
        Raw["2.4M Emails/Slack/Excel<br/>(Exposes veryHidden & White Text)"] --> CAL["📈 TAR 2.0 Continuous Active Learning<br/>+ Binomial Elusion Stopping Proof<br/>(96.4% Recall @ 15.5% Reviewed)"]
    end

    subgraph Priv["2. Counsel Graph & Family Closure Lock"]
        CAL --> Graph["⚖️ Matter Counsel Graph + Thread Tree<br/>Catches Non-Lawyer Relayed Advice<br/>& Dual-Hat GC Primary Purpose"]
        Graph --> Lock["🔒 Parent-Child & 95% MinHash Lock<br/>Locks All 14 Thread Variants Together"]
    end

    subgraph Prod["3. 300-DPI Raster Burn & FRCP 26(b)(5) Log"]
        Lock -->|"Responsive with Redactions"| Burn["🛡️ 300-DPI Bitmap Raster Burn<br/>Strips Vector Text + Post-Burn OCR == 0"]
        Lock -->|"Withheld Privileged (42K Docs)"| Log["📜 FRCP Rule 26(b)(5) Privilege Logger<br/>Non-Waiving Statutory Basis Descriptions"]
    end
```

### 3.3 End-to-End Sequence: Intercepting Relayed Counsel Advice & Verifying Zero-Leak Raster Redaction

```mermaid
sequenceDiagram
    autonumber
    participant Corpus as 2.4M Custodian Corpus (DOC-881920)
    participant CAL as TAR 2.0 CAL + Counsel Graph Engine
    participant Lock as Family & 95% Near-Dup Closure Lock
    participant Burn as 300-DPI Raster Burn & FRCP Logger
    participant Court as Opposing Counsel & Federal Court

    Corpus->>CAL: Ingest VP-to-VP email: "Per Sarah's guidance yesterday, pause API launch until patent indemnity is revised"
    CAL->>CAL: Trace email thread ancestry & Matter Counsel Graph -> resolve "Sarah" to External Patent Counsel Sarah Lin, Esq.
    CAL->>Lock: Flag implicit relay of legal advice as WITHHOLD_PRIVILEGED (Send to a human)
    Lock->>Lock: Lock parent-child attachment family & all 14 95%-MinHash near-duplicate thread variants from production!
    Burn->>Burn: Rasterize 45,000 partially redacted PDFs to 300-DPI bitmaps; run post-burn OCR over boxes -> 0 readable bytes
    Burn-->>Court: ✅ Deliver Elusion Affidavit (96.4% recall) + zero-leak Bates production + FRCP Rule 26(b)(5) Privilege Log!
```

---

## ④ 📊 4-Tier Architecture Decision Matrix

| Evaluation Dimension | 🟥 Tier 1: Keywords + Linear Human | 🟨 Tier 2: Standalone Zero-Shot LLM | 🟦 Tier 3: Legacy `TAR 1.0` | 🟩 Tier 4: `TAR 2.0 CAL` + Counsel Graph + Raster |
|---|---|---|---|---|
| **Responsive Document Recall** | `58.0%` ✕ (misses non-keyword docs) | `76.5%` (uncalibrated cutoff) | `81.0%` | 🏆 **`96.4%` (`TAR 2.0 CAL` + binomial elusion proof)** |
| **Inadvertent Privilege Waiver** | `3.4%` ✕ (human fatigue slips) | `6.8%` ✕ (misses non-lawyer relays) | `2.1%` ✕ (split thread coding) | 🏆 **`0.0%` (Counsel Graph + family/near-dup lock)** |
| **Redaction Text-Layer Leak Rate** | `2.8%` ✕ (vector black box overlay) | `4.1%` ✕ (hidden Excel caches) | `1.4%` ✕ | 🏆 **`0.0%` (`300-DPI` bitmap burn + post-burn OCR)** |
| **Human Review Volume Reduction** | `0.0%` ✕ (reads 100% of hits) | `65.0%` *(no court stopping proof)* | `58.0%` | 🏆 **`84.5%` (stops defensibly at 15.5% reviewed)** |
| **Monthly Review & Risk TCO** | `$1,150,000 / mo` | `$740,000 / mo` | `$495,000 / mo` | **`$138,000 / mo` 🏆 (-88.0%)** |

---

## ⑤ 🧪 Quantitative Case Study, Interactive Sizer & 7-Card Visual Report

- **Interactive HTML Sizer**: 👉 [Open Enterprise Legal E-Discovery, Privilege Review & Redaction Sizer](../reports/enterprise-legal-ediscovery-privilege-review-tar-cal-redaction-agent-sizer.html)
- **7-Card 4:5 Visual Benchmark Report**: 👉 [Open 7-Card Visual Benchmark Deck](../reports/enterprise-legal-ediscovery-privilege-review-tar-cal-redaction-agent-selection-cards.html)
- **Deterministic Verification**:
  ```bash
  python experiments/enterprise_legal_ediscovery_privilege_review_tar_cal_redaction_agent_selection/run.py --check
  python -m unittest tests/test_enterprise_legal_ediscovery_privilege_review_tar_cal_redaction_agent_selection.py
  ```

---

## ⑥ 🛠️ Production Readiness & Architecture Review Checklist

- [ ] **Replace Pure Boolean Keyword Culled Review with `TAR 2.0 Continuous Active Learning (CAL)` and Stratified Binomial Elusion Testing**: Establish court-defensible recall (`>=95%`) before halting human review.
- [ ] **Resolve All Communicators and Mentioned Names Against a Matter Entity-Counsel Directory Graph and Email-Thread Ancestry**: Catch implicit relays of legal advice between non-lawyer business executives.
- [ ] **Enforce Deterministic Privilege Closure Across Parent-Child Document Families and `95% MinHash` Near-Duplicates**: Never allow a privileged attachment or thread branch to be produced via a differently coded variant.
- [ ] **Burn All Redacted Pages into Flattened `300-DPI` Bitmaps and Strip Native `OOXML` Hidden Layers (`veryHidden` Sheets, Pivot Caches, White-on-White Text)**: Verify zero extractable characters via post-burn OCR before Bates numbering.
- [ ] **Generate `FRCP Rule 26(b)(5)` Privilege Logs with a Non-Waiving Substance Filter**: State author, recipient, counsel role, and statutory privilege basis without leaking the underlying legal advice.
