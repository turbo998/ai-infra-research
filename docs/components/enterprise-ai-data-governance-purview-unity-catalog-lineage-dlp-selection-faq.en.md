# Enterprise AI Data Governance, Sensitivity Label Inheritance (`Microsoft Purview + Databricks Unity Catalog`), `AI-BOM` Corpus Lineage & `15-Minute` Tombstone Unlearning Selection Guide: How to Prevent RAG Chunk Label Stripping and `$40K` GDPR Model Retraining?

> **Language / 语言**: **English (Current)** | [中文](./enterprise-ai-data-governance-purview-unity-catalog-lineage-dlp-selection-faq.md)  
> **Scope**: End-to-end production reference architecture and governance guide for Cloud & AI Solution Architects, Chief Data Officers (CDOs), CISOs, and AI Compliance Leads governing unstructured M365 documents (`Microsoft Purview`), structured Lakehouse tables (`Databricks Unity Catalog / Snowflake Horizon`), RAG vector indexes, and SFT/LoRA training corpora.  
> **Provenance**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`). Deterministic evaluation model sealed in [`experiments/enterprise_ai_data_governance_purview_unity_catalog_lineage_dlp_selection/results.json`](../../experiments/enterprise_ai_data_governance_purview_unity_catalog_lineage_dlp_selection/results.json).

![Enterprise AI Data Governance: Purview + Unity Catalog + Lineage & Unlearning Summary](../../assets/svg/enterprise-ai-data-governance-purview-unity-catalog-lineage-dlp-selection-faq-summary.svg)

---

## ① 📌 Executive Summary (3-Minute Architect Verdict)

More than 80% of enterprise AI projects that stall before production are blocked not by model quality, but by **CISO, Legal, and Data Governance sign-off**. Across an enterprise estate of **25 million unstructured documents (`SharePoint / OneDrive / Teams / PDFs`), 12,000 Lakehouse tables, and 18 fine-tuned `LoRA` adapters**, three data-governance failures routinely trigger compliance incidents:
1. **Trap #1: The "RAG Chunk Label-Stripping" Vulnerability (Tier 1, `19.4%` Sensitive Data Over-Exposure Rate)**: A board M&A presentation is tagged `L4 Strictly Confidential` in `Microsoft Purview`, but when a naive Python ETL script chunks the PDF into 512-token vectors, it strips the file's `MIP sensitivity label` and `Entra security group SIDs`—making restricted board secrets searchable by any employee!
2. **Trap #2: Baking Mutable Customer PII or Revocable Records into `SFT / LoRA` Model Weights (Tier 1, `720-Hour` Deletion SLA & `$40,000` Retraining per GDPR Request)**: Deleting a record from a vector database takes milliseconds, but once customer names or private contract terms are baked into neural network weights during fine-tuning, honoring a single **GDPR Article 17 Right-to-be-Forgotten** request forces a full model retrain!
3. **Trap #3: Governing Only Structured Tables (`Unity Catalog`) While Leaving Unstructured Documents Unlabeled (Tier 3, `4.2%` Exposure Rate, `180–640` Audit Prep Hours/Quarter)**: Without federated lineage linking `Source Document SHA-256 -> Vector Chunk -> Prompt / Fine-Tuned Adapter`, compliance teams spend hundreds of hours manually reconstructing training provenance for EU AI Act and ISO/IEC 42001 audits.

Evaluated across **25M unstructured documents, 12,000 Lakehouse tables, and 45,000 monthly GDPR deletion / ACL change events**, the four AI data governance architectures compare as follows:

| Architectural Tier | Classification, Lineage & Unlearning Stack | Sensitive Exposure Rate | `GDPR` Deletion SLA | Lineage Coverage (`AI-BOM`) | Audit Prep Hrs / Qtr | Monthly Governance TCO | Production Verdict |
|---|---|---|---|---|---|---|---|
| **🟥 Tier 1: Unclassified Raw Dump (`Vector DB + SFT`)** | Dumps folders into RAG & raw tickets into SFT weights | `19.4%` ✕ *(strips MIP labels)* | `720.0 hrs` ✕ *(forces retrain)* | `0.0%` ✕ | `640 hrs/qtr` ✕ | `$215,000 / mo` ✕ *(incl. retrain/fines)* | ⛔ Blocked by CISO; severe data leak & GDPR liability |
| **🟨 Tier 2: Manual Folder Whitelists + Full Rebuild** | Manual legal review; rebuilds whole HNSW index on delete | `6.8%` | `168.0 hrs (7d)` ✕ | `35.0%` | `420 hrs/qtr` | `$148,000 / mo` | ⛔ 7-day deletion lag & expensive index rebuilds |
| **🟦 Tier 3: Siloed Lakehouse Catalog Only (`Unity`)** | Governs SQL tables, but PDFs lose Purview labels in RAG | `4.2%` ✕ *(unstructured gap)* | `48.0 hrs` | `62.0%` | `180 hrs/qtr` | `$92,000 / mo` | ⚠️ Misses unstructured M365 document sensitivity tags |
| **🟩 Tier 4: Federated `Purview + Unity Catalog` + Chunk Label Inheritance + `15m` Tombstone CDC + Zero-PII SFT Gate** | **Stamps `[parent_doc_id, mip_label, acl_sids]` on every chunk + `15m` Tombstone delete + blocks PII from SFT** | **`0.0%` 🏆** *(zero leaks)* | **`0.25 hrs (15m)` 🏆** *(tombstone purge)* | **`100.0%` 🏆** *(SHA-256 graph)* | **`12 hrs/qtr` 🏆** *(-98.1% labor)* | **`$38,000 / mo` 🏆** *(-82.3% vs Tier 1)* | 🌟 **Recommended Enterprise AI Data Governance Standard** |

> 💡 **Golden Rules for Enterprise AI Data Governance Architects**:
> 1. **Chunk-Level Sensitivity Inheritance**: **"Every 512-token vector chunk must inherit its parent document's `Purview MIP Sensitivity Label (L1–L4)` and `Entra Security Group SIDs`!"** Block `L4 (Board / Credentials)` at ingestion and enforce pre-filter SID intersection at query time (`0.0%` over-exposure).
> 2. **The `RAG vs. SFT` Governance Firewall**: **"Mutable, revocable, or customer-identifiable (PII) data must stay 100% in `RAG / Lakehouse` and NEVER enter `SFT / LoRA` model weights!"** Enforce an automated `Purview / Presidio Zero-PII Gate` before any fine-tuning job starts, and honor GDPR deletions in **`15 minutes`** via `parent_doc_id` Tombstone CDC.
> 3. **Federated Dual-Catalog Architecture**: Use **`Microsoft Purview`** for unstructured M365 documents/emails and **`Databricks Unity Catalog`** for structured Lakehouse tables (`ABAC` row/column masks) and `MLflow` models, linked via `OpenLineage` (`100%` AI-BOM coverage).

---

## ② 🎯 Customer & Architect Deep-Dive Q&A (3 Chunk Label Stripping, GDPR SFT Retraining & Purview-vs-Unity Pain Points)

### Q1: How do we prevent a confidential SharePoint document from losing its security classification when split into RAG vector chunks?
- **Mandatory Chunk Metadata Sealing (`[parent_doc_id, mip_label_id, allowed_sids]`)**:
  - Standard document parsers (`PyMuPDF / Docling`) extract raw text and discard file-system metadata.
  - Tier 4 wraps the parser in a **Metadata Inheritance Gate**: (1) if `mip_label == L4_STRICT_CONFIDENTIAL`, quarantine the file from general RAG indexing; (2) for `L1–L3` files, stamp every generated vector chunk with its parent's immutable `parent_doc_id`, `doc_version_sha256`, `mip_label`, and `allowed_entra_sids`—ensuring vector search enforces the exact same access control as SharePoint!

### Q2: How do we honor a customer's `GDPR Right-to-be-Forgotten` deletion request in `15 minutes` without retraining our fine-tuned models?
- **Pre-SFT Zero-PII Quarantine Gate + `15-Minute` `parent_doc_id` Tombstone CDC**:
  - **Before Training**: A CI/CD governance gate scans every `SFT / LoRA` JSONL dataset with `Presidio + Purview DLP` and blocks training if a single customer SSN, email, or private account ID is present. Model weights learn *task formatting and domain reasoning*, never *customer facts*.
  - **At Deletion Time**: Because all customer facts live in `Unity Catalog` tables and `RAG` vector chunks indexed by `parent_doc_id / customer_id`, a Kafka Tombstone CDC event executes `DELETE WHERE parent_doc_id = X` across the vector index and Redis semantic cache in **`0.25 hours (15 minutes)`** without rebuilding the HNSW graph!

### Q3: When should an enterprise use `Microsoft Purview` vs. `Databricks Unity Catalog` for AI governance?
- **Complementary Domain Authority with Bidirectional Lineage**:
  - **`Microsoft Purview`**: Governs **unstructured enterprise knowledge** (`SharePoint`, `OneDrive`, `Teams`, `Exchange`, PDFs/Office docs), `MIP L1–L4` sensitivity labels, and M365 Copilot / Azure AI Foundry DLP policies;
  - **`Databricks Unity Catalog`**: Governs **structured & semi-structured Lakehouse data** (`Delta / Iceberg` tables with dynamic `ABAC` row filters and column masks), Feature Stores, Vector Search indexes, and `MLflow` model/adapter registries;
  - **Federated `AI-BOM` Lineage**: Syncing `OpenLineage` events between `Purview` and `Unity Catalog` gives auditors a 1-click cryptographic graph from `Source File SHA-256 -> Vector Chunk / Training Dataset -> Model Version` (`12 audit hours/quarter` vs. `640 hours`).

---

## ③ 🧭 Visual Architecture Diagrams (Decision Tree + Federated `Purview + Unity` Topology + `45s` Label Upgrade Sequence)

### 3.1 Enterprise AI Data Governance & Corpus Lineage Selection Decision Tree

```mermaid
flowchart TD
    A["Enterprise AI Data Governance & Lineage Selection<br/>(25M Unstructured Docs + 12K Lakehouse Tables + 18 LoRA Models)"] --> B{"Are mutable customer PII or revocable docs allowed into SFT weights?"}
    B -- "Raw tickets/docs fed directly into SFT/LoRA (Tier 1)" --> C["⛔ $40K Retraining per GDPR Delete & 720-Hr Lag!<br/>Customer PII baked into neural network weights"]
    B -- "Strict Firewall: Mutable/PII in RAG Only; Zero-PII SFT Gate" --> D{"How are unstructured M365 docs and structured Lakehouse tables governed?"}
    D -- "Manual folder whitelists + full HNSW index rebuilds (Tier 2)" --> E["⛔ 168-Hour (7-Day) Deletion Lag & 420 Audit Hrs/Qtr!"]
    D -- "Lakehouse Unity Catalog only; PDF chunks lose MIP labels (Tier 3)" --> F["⚠️ 4.2% Sensitive Exposure!<br/>SharePoint PDFs lose classification tags during RAG chunking"]
    D -- "Federated Purview (M365) + Unity Catalog (Lakehouse) + 15m CDC (Tier 4)" --> G["🟩 Tier 4: Chunk MIP Inheritance + 15m Tombstone CDC + 100% AI-BOM 🏆<br/>(0.0% Over-Exposure | 0.25h GDPR Purge | 12 Audit Hrs/Qtr | -82.3% TCO)"]
    style G fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#f8fafc
    style C fill:#7f1d1d,stroke:#ef4444,color:#fecaca
    style E fill:#7f1d1d,stroke:#ef4444,color:#fecaca
```

### 3.2 Tier 4 Federated `Microsoft Purview + Databricks Unity Catalog` & Tombstone Unlearning Architecture

```mermaid
flowchart LR
    subgraph Estate["1. Dual Enterprise Data Estate"]
        Docs["📄 25M Unstructured M365/PDF Docs<br/>Governed by Microsoft Purview<br/>(MIP L1–L4 Sensitivity Labels & SIDs)"]
        Tables["🗄️ 12,000 Lakehouse Tables<br/>Governed by Databricks Unity Catalog<br/>(ABAC Row Filters & Column Masks)"]
    end

    subgraph Gates["2. Ingestion Seal & SFT Zero-PII Firewall"]
        Docs -->|Block L4 at Gate| Seal["🏷️ Chunk Metadata Inheritance Proxy<br/>Stamps [parent_doc_id, mip_label, allowed_sids]"]
        Docs & Tables --> SFT["🛡️ Pre-SFT PII & Secret Quarantine Gate<br/>Blocks training if PII count > 0"]
    end

    subgraph Lineage["3. 15-Min Tombstone CDC & 100% AI-BOM Lineage"]
        Seal --> VDB["🔍 ACL-Filtered Vector DB + Semantic Cache<br/>(15-Min Tombstone Purge by parent_doc_id)"]
        SFT --> BOM["📦 Unity Catalog + Purview OpenLineage<br/>(100% SHA-256 Source -> Chunk -> Model Graph)"]
    end
```

### 3.3 End-to-End Sequence: `45-Second` Cascade Purge When a SharePoint Document Is Upgraded from `L2 Internal` to `L4 Board Only`

```mermaid
sequenceDiagram
    autonumber
    participant Legal as Corporate Secretary
    participant Purview as Microsoft Purview / SharePoint
    participant CDC as Kafka Tombstone CDC Stream
    participant VDB as Enterprise Vector DB (Indexed by parent_doc_id)
    participant Cache as Redis Semantic Cache

    Legal->>Purview: 1. 14:00:00 Upgrade `M&A_Draft.docx` (doc_id=D-901) from `L2_INTERNAL` to `L4_BOARD_RESTRICTED`
    Purview->>CDC: 2. Emit Graph Webhook event `{doc_id: "D-901", new_label: "L4_BOARD"}`
    par t = 15s..45s: Parallel Tombstone Purge across Vector Index & Semantic Cache
        CDC->>VDB: 3. Execute `DELETE FROM rag_chunks WHERE parent_doc_id = 'D-901'` (purges all 42 child chunks)
        CDC->>Cache: 4. Invalidate all semantic cache entries referencing `parent_doc_id = 'D-901'`
    end
    Note over Legal,Cache: By 14:00:45 (45s elapsed): 0.0% residual exposure across all RAG and Copilot endpoints!
```

---

## ④ 📊 4-Tier Architecture Decision Matrix

| Evaluation Dimension | 🟥 Tier 1: Unclassified Raw Dump | 🟨 Tier 2: Manual Whitelist + Rebuild | 🟦 Tier 3: Lakehouse Catalog Only | 🟩 Tier 4: Federated `Purview + Unity` + `15m CDC` |
|---|---|---|---|---|
| **Sensitive Over-Exposure Rate** | `19.4%` ✕ (strips MIP labels) | `6.8%` | `4.2%` ✕ (unlabeled PDF chunks) | 🏆 **`0.0%` (`L4` gate + chunk-level `MIP/SID` inheritance)** |
| **`GDPR` Deletion / Revocation SLA** | `720.0 hrs` ✕ (model retrain) | `168.0 hrs (7d)` ✕ | `48.0 hrs` | 🏆 **`0.25 hrs (15 mins)` (`parent_doc_id` tombstone purge)** |
| **End-to-End `AI-BOM` Lineage** | `0.0%` ✕ (no provenance) | `35.0%` | `62.0%` | 🏆 **`100.0%` (`SHA-256` source -> chunk -> model graph)** |
| **Compliance Audit Prep Hours/Qtr** | `640 hrs/qtr` ✕ | `420 hrs/qtr` | `180 hrs/qtr` | 🏆 **`12 hrs/qtr` (automated ISO 42001 / EU AI Act export)** |
| **Monthly Governance & Audit TCO** | `$215,000 / mo` ✕ | `$148,000 / mo` | `$92,000 / mo` | **`$38,000 / mo` 🏆 (-82.3% vs Tier 1)** |

---

## ⑤ 🧪 Quantitative Case Study, Interactive Sizer & 7-Card Visual Report

- **Interactive HTML Sizer**: 👉 [Open Enterprise AI Data Governance, Purview/Unity Catalog & Lineage Sizer](../reports/enterprise-ai-data-governance-purview-unity-catalog-lineage-dlp-sizer.html)
- **7-Card 4:5 Visual Benchmark Report**: 👉 [Open 7-Card Visual Benchmark Deck](../reports/enterprise-ai-data-governance-purview-unity-catalog-lineage-dlp-selection-cards.html)
- **Deterministic Verification**:
  ```bash
  python experiments/enterprise_ai_data_governance_purview_unity_catalog_lineage_dlp_selection/run.py --check
  python -m unittest tests/test_enterprise_ai_data_governance_purview_unity_catalog_lineage_dlp_selection.py
  ```

---

## ⑥ 🛠️ Production Readiness & Architecture Review Checklist

- [ ] **Stamp `[parent_doc_id, doc_version_sha256, mip_label_id, allowed_sids]` on Every RAG Vector Chunk**: Never allow document parsers to strip `Microsoft Purview` sensitivity labels, and block `L4 Strictly Confidential` files at the ingestion gate.
- [ ] **Enforce a Zero-PII Quarantine Gate Before Every `SFT / LoRA` Fine-Tuning Run**: Keep mutable customer records and revocable documents exclusively in `RAG / Lakehouse` so GDPR deletions never require model retraining.
- [ ] **Wire `15-Minute` Tombstone CDC (`DELETE WHERE parent_doc_id = X`) Across Vector Indexes and Semantic Caches**: Instantly purge deleted or re-classified documents without full HNSW index rebuilds.
- [ ] **Federate `Microsoft Purview` (Unstructured M365 Docs) with `Databricks Unity Catalog` (Structured Lakehouse `ABAC` & `MLflow` Models)**: Eliminate governance blind spots between office documents and data warehouse tables.
- [ ] **Maintain an Immutable `SHA-256 AI-BOM` OpenLineage Graph**: Record exact source document hashes for every retrieved RAG chunk and fine-tuned adapter version.
