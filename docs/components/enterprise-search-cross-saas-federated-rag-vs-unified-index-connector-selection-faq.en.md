# Enterprise Cross-SaaS Workplace Search & Copilot (`SharePoint + Confluence + Jira + Slack + Salesforce`), Live Federated API Fan-Out vs. Unified Connector Index Hybrid Routing Architecture Selection Guide

> **Language / 语言**: **English (Current)** | [中文](./enterprise-search-cross-saas-federated-rag-vs-unified-index-connector-selection-faq.md)  
> **Scope**: End-to-end production reference architecture and sizing playbook for Cloud & AI Solution Architects, Digital Workplace Leads, and Enterprise Search Engineers building cross-silo internal search and employee copilots (`15,000+` employees across `8+` SaaS systems).  
> **Provenance**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`). Deterministic evaluation model sealed in [`experiments/enterprise_search_cross_saas_federated_rag_vs_unified_index_connector_selection/results.json`](../../experiments/enterprise_search_cross_saas_federated_rag_vs_unified_index_connector_selection/results.json).

![Enterprise Cross-SaaS Workplace Search Federated vs Unified Index Summary](../../assets/svg/enterprise-search-cross-saas-federated-rag-vs-unified-index-connector-selection-faq-summary.svg)

---

## ① 📌 Executive Summary (3-Minute Architect Verdict)

When building an enterprise-wide workplace search portal and employee copilot spanning **`8+` heterogeneous SaaS silos (`SharePoint`, `Confluence`, `Jira`, `ServiceNow`, `Slack/Teams`, `Salesforce`, `GitHub`, `Workday` — `15,000` employees, `25M` items, `4.5M` monthly searches)**, architects face a critical design choice: **Live Federated API Fan-Out vs. Central Unified Connector Index**. Three common architectures fail at enterprise scale:
1. **Trap #1: 100% Live Federated API Fan-Out Across All 8 SaaS Silos on Every Query (`Tier 1`, `4,200ms` P99 Search Latency, `22.5%` Upstream SaaS `HTTP 429` Rate-Limit Drops, `68.0%` Cross-Silo `NDCG@10`, `$162,000/month` TCO)**:
   - Fanning out every query to 8 external REST APIs means P99 latency equals the **slowest SaaS API (`4.2s`)**!
   - Legacy SaaS keyword search APIs lack semantic embeddings and return incomparable raw scores—while morning peak traffic exhausts tenant-wide `Jira / ServiceNow / Salesforce` API rate limits (`22.5%` `HTTP 429` failures)!
2. **Trap #2: 100% Central Unified Index Copy of Everything Including Live ERP/HR Tables (`Tier 2`, `18.0%` Stale Live Ticket/Deal Answers, Blocked by CISO over Copying Sensitive `Workday/Salesforce` Records, `$145,000/month` TCO)**.
3. **Trap #3: Multi-Turn Sequential Agentic Tool-Calling Loop (`Tier 3`, `7,800ms` Latency as the LLM Calls Silos One by One, `$178,000/month` TCO)**.

Evaluated across **15,000 employees, 8 SaaS silos, 25 million documents/records, and 4.5 million monthly searches**, the four workplace search architectures compare as follows:

| Architecture Tier | Indexing, Federation, ACL & Ranking Stack | Cross-Silo `NDCG@10` | Search `P99` Latency | Upstream SaaS `429` Rate | Monthly Total TCO | Production Verdict |
|---|---|---|---|---|---|---|---|
| **🟥 Tier 1: 100% Live Federated API Fan-Out** | Queries all 8 SaaS native REST search APIs on every search | `68.0%` ✕ *(keyword-only SaaS APIs)* | `4,200.0 ms` ✕ *(bound by slowest API)* | `22.5%` ✕ *(exhausts SaaS quotas)* | `$162,000 / mo` ✕ *(quota + wait waste)* | ⛔ Fails on both latency (`4.2s`) and SaaS API rate limits |
| **🟨 Tier 2: 100% Central Unified Index Copy** | Replicates all docs + live HR/CRM tables into vector DB | `89.5%` | `210.0 ms` | `2.4%` | `$145,000 / mo` ✕ *(18% stale state)* | ⛔ Serves stale live ticket/deal state & violates HR data minimization |
| **🟦 Tier 3: Sequential Multi-Turn Agentic Tool Loop** | LLM calls SaaS MCP tools sequentially across 3–4 turns | `86.0%` | **`7,800.0 ms` ✕** *(7.8s lag!)* | `8.5%` | `$178,000 / mo` ✕ *(multi-turn tokens)* | ⛔ 7.8s sequential loop is unusable for interactive search |
| **🟩 Tier 4: `85% / 15%` Split-Plane (`Unified Hybrid Index` for Docs + `Intent-Gated Federated API` for Live Records + `Cross-Encoder`)** | **`45ms` unified index for `SharePoint/Confluence/Slack` + `10ms` router to live `Jira/Workday/CRM` APIs + `28ms` Cross-Encoder** | **`96.4%` 🏆** *(+28.4 pts vs Tier 1)* | **`380.0 ms` 🏆** *(11.1x faster!)* | **`0.0%` 🏆** *(zero API storms)* | **`$24,000 / mo` 🏆** *(-85.2% vs Tier 1)* | 🌟 **Recommended Enterprise Cross-SaaS Search Standard** |

> 💡 **Golden Rules for Enterprise Cross-SaaS Workplace Search**:
> 1. **The `85% / 15%` Split-Plane Architecture**:
>    - **Plane A (`85%` Unstructured Knowledge — `SharePoint`, `Confluence`, `Slack`, `GitHub Docs`)**: Ingest via incremental Webhook/Delta connectors into a **Unified Hybrid (`BM25 + Dense Vector`) Index**. Delivers **`45ms` semantic retrieval** with **zero runtime SaaS API calls**!
>    - **Plane B (`15%` Live Transactional State & High-Sensitivity Records — `Workday PTO/Payroll`, `Salesforce Live Opportunity Stage`, `Jira/ServiceNow Active Blockers`)**: Never copy into the vector DB! A **`10ms` Source-Intent Classifier** triggers a targeted, single-silo OAuth API call **only** when the query requests live transactional state (`0.0%` SaaS `429` throttling, `100%` real-time freshness).
> 2. **Pre-Flattened Identity Graph ACL Bitmaps (`Entra ID + Okta + SaaS Groups`)**: Resolve cross-SaaS user identities into an in-memory **Roaring Bitmap** of permitted group IDs so Plane-A ACL filtering executes in **`2.0ms`** inside the vector query.
> 3. **Unified Cross-Encoder Score Calibration (`Plane C`, `28ms`)**: Re-score candidate snippets from Plane A and Plane B together through a single `BGE-Reranker-v2-M3` Cross-Encoder to normalize incomparable source scores (`96.4%` `NDCG@10`)!

---

## ② 🎯 Customer & Architect Deep-Dive Q&A (3 Why Pure Federation Is Slow & Blind, Protecting HR/CRM Data & Score Normalization Pain Points)

### Q1: Why does `100%` Live Federated Search (`Tier 1`) achieve only `68.0%` `NDCG@10` while suffering `4,200ms` P99 latency and `22.5%` `HTTP 429` errors?
- **Tail-Latency Amplification & Keyword-Only SaaS Endpoints**:
  - Querying 8 SaaS APIs in parallel means your P99 latency is dictated by the slowest upstream API (`4.2s`), and `4.5M` monthly searches multiply into **`36,000,000` outbound SaaS API calls/month**—blowing through Jira and ServiceNow tenant rate limits at 9:30 AM!
  - Furthermore, most SaaS REST search APIs only run basic keyword matching over full documents (not chunk-level semantic embeddings), missing `32%` of conceptual queries.

### Q2: How do we satisfy CISO data-minimization rules for `Workday` (HR) and `Salesforce` (CRM) without sacrificing `45ms` semantic search across `SharePoint` and `Confluence`?
- **Routing by Data Volatility & Sensitivity (`Plane A` vs. `Plane B`)**:
  - Unstructured collaboration docs (`SharePoint`, `Confluence`, `Slack`, `Engineering RFCs`) belong in **Plane A (Unified Index)** where chunking and embeddings are essential.
  - Structured Systems of Record (`Workday`, `Salesforce`, `SAP`, `Jira live sprint fields`) stay in **Plane B (Zero-Copy Targeted Federation)**—queried on-demand via the user's OAuth token only on the `15%` of queries classified as transactional lookups!

### Q3: How do we merge results when Plane A returns a Confluence page with RRF score `0.032` and Plane B returns a live Jira ticket with REST API score `18.4`?
- **Plane C: Unified Cross-Encoder Reranking (`28ms`)**:
  - Raw retrieval scores across different engines are mathematically incomparable. Converting both the Confluence chunk and the Jira ticket summary into uniform text cards and scoring all `Top-25` candidates against the user query in a single **`TEI Cross-Encoder` batch (`28ms`)** maps every result onto a calibrated `[0.0, 1.0]` probability scale (`96.4%` `NDCG@10`)!

---

## ③ 🧭 Visual Architecture Diagrams (Decision Tree + `85/15` Split-Plane Topology + Cross-Plane Execution Sequence)

### 3.1 Enterprise Cross-SaaS Workplace Search Decision Tree

```mermaid
flowchart TD
    A["Enterprise Cross-SaaS Workplace Search Selection<br/>(15,000 Employees | 8 SaaS Silos | 25M Items | 4.5M Searches/Mo)"] --> B{"How do you search across unstructured docs & live SaaS systems?"}
    B -- "100% live federated API fan-out to all 8 silos (Tier 1)" --> C["⛔ 4,200ms P99 Latency + 22.5% SaaS HTTP 429 Drops!<br/>Poor keyword-only SaaS recall (68.0% NDCG@10, $162K/mo TCO)"]
    B -- "100% central index copy including live ERP/HR tables (Tier 2)" --> D["⛔ 18.0% Stale Live Ticket/Deal Status!<br/>CISO blocks copying sensitive Workday/Salesforce tables ($145K/mo)"]
    B -- "Sequential multi-turn agentic SaaS tool-calling loop (Tier 3)" --> E["⛔ 7,800ms (7.8s) Multi-Turn Latency!<br/>Far too slow & expensive for interactive workplace search ($178K/mo)"]
    B -- "85% Unified Hybrid Index (Docs) + 15% Targeted Federated API + Cross-Encoder (Tier 4)" --> F["🟩 Tier 4: 85/15 Split-Plane Index + Targeted Federation + Cross-Encoder 🏆<br/>(96.4% NDCG@10 | 380ms P99 | 0.0% SaaS 429s | $24K/mo, -85.2% TCO)"]
    style F fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#f8fafc
    style C fill:#7f1d1d,stroke:#ef4444,color:#fecaca
    style D fill:#7f1d1d,stroke:#ef4444,color:#fecaca
    style E fill:#7f1d1d,stroke:#ef4444,color:#fecaca
```

### 3.2 Tier 4 `85/15` Split-Plane Hybrid Index, Targeted Federation & Unified Cross-Encoder Architecture

```mermaid
flowchart LR
    subgraph Gate["1. 10ms Intent Router & Identity Graph"]
        User["👤 Employee Query<br/>(Entra ID OAuth)"] --> Route["🧭 10ms Source Intent Router<br/>Unstructured Docs (85%) vs.<br/>Live Transactional Silo (15%)"]
        User --> ACL["🔐 Unified Identity Graph<br/>Pre-flattened Group ACL Bitmap"]
    end

    subgraph PlaneA["2A. Plane A: Unified Hybrid Index (85% Traffic, 45ms)"]
        Route & ACL --> Idx[("🗂️ Unified BM25 + Vector Index<br/>(SharePoint / Confluence / Slack / GitHub)<br/>Webhook CDC + 2ms ACL Bitmap<br/>0 runtime SaaS API calls!")]
    end

    subgraph PlaneB["2B. Plane B: Targeted Live Federation (15% Traffic, 280ms)"]
        Route -- "Triggered ONLY on live record intent" --> Live["⚡ Single-Silo OAuth Live API<br/>(Workday PTO / Salesforce Deal / Jira)<br/>Zero-copy | 0.0% 429 storms"]
    end

    subgraph PlaneC["3. Plane C: Score Calibration & Reranking (28ms)"]
        Idx & Live --> CE["🎯 TEI Cross-Encoder Fusion<br/>Normalizes Plane A + B scores -> 96.4% NDCG@10!"]
    end
```

### 3.3 End-to-End Sequence: Resolving a Cross-Silo Engineering + Live Jira Blocker Query in `318ms`

```mermaid
sequenceDiagram
    autonumber
    participant Eng as Engineer
    participant GW as Search Gateway (10ms Intent + ACL Bitmap)
    participant Idx as Plane A: Unified Index (Confluence/Slack)
    participant Jira as Plane B: Live Jira REST API
    participant CE as Plane C: TEI Cross-Encoder & Generator

    Eng->>GW: 1. Query: `"What caused the Acme SSO outage and what is the current status of the Jira P1 blocker?"`
    GW->>GW: 2. `10ms`: Classify intent = `[Unstructured_Docs, Live_Jira]` & resolve user's cross-SaaS Group ACL bitmap
    par Parallel Split-Plane Retrieval (max(45ms, 280ms) = 280ms; skips Workday/Salesforce/ServiceNow!)
        GW->>Idx: 3A. Search Unified Hybrid Index (`45ms` -> returns Confluence `SAML X.509 Postmortem` + Slack thread)
        GW->>Jira: 3B. Execute targeted OAuth Jira API call (`280ms` -> returns `JIRA-8812: Awaiting customer cert upload`)
    end
    Idx & Jira-->>CE: 4. Merge Plane-A chunks and Plane-B live ticket card
    CE->>Eng: 5. `28ms` Cross-Encoder reranking + cited streaming answer (`318ms` total retrieval, `0` wasted API calls)!
```

---

## ④ 📊 4-Tier Architecture Decision Matrix

| Evaluation Dimension | 🟥 Tier 1: 100% Live Federated | 🟨 Tier 2: 100% Central Copy | 🟦 Tier 3: Sequential Agent Loop | 🟩 Tier 4: `85/15` Split-Plane + `Cross-Encoder` |
|---|---|---|---|---|
| **Cross-Silo `NDCG@10` Precision** | `68.0%` ✕ (no semantic vectors) | `89.5%` | `86.0%` | 🏆 **`96.4%` (unified embeddings + Cross-Encoder calibration)** |
| **Search `P99` Latency** | `4,200.0 ms` ✕ (slowest API lag) | `210.0 ms` | `7,800.0 ms` ✕ (multi-turn lag) | 🏆 **`380.0 ms` (11.1x faster than Tier 1)** |
| **Upstream SaaS `HTTP 429` Drops** | `22.5%` ✕ (rate-limit storms) | `2.4%` | `8.5%` | 🏆 **`0.0%` (eliminates 85% of unnecessary API fan-out)** |
| **Live Record Freshness & CISO Fit** | Real-time, but unusable | `18.0%` stale + CISO block | Real-time, too slow | 🏆 **`100%` live & zero-copy for sensitive HR/CRM tables** |
| **15K-Employee Monthly TCO** | `$162,000 / mo` ✕ | `$145,000 / mo` ✕ | `$178,000 / mo` ✕ | **`$24,000 / mo` 🏆 (-85.2% vs Tier 1)** |

---

## ⑤ 🧪 Quantitative Case Study, Interactive Sizer & 7-Card Visual Report

- **Interactive HTML Sizer**: 👉 [Open Enterprise Cross-SaaS Workplace Search Sizer](../reports/enterprise-search-cross-saas-federated-rag-vs-unified-index-connector-sizer.html)
- **7-Card 4:5 Visual Benchmark Report**: 👉 [Open 7-Card Visual Benchmark Deck](../reports/enterprise-search-cross-saas-federated-rag-vs-unified-index-connector-selection-cards.html)
- **Deterministic Verification**:
  ```bash
  python experiments/enterprise_search_cross_saas_federated_rag_vs_unified_index_connector_selection/run.py --check
  python -m unittest tests/test_enterprise_search_cross_saas_federated_rag_vs_unified_index_connector_selection.py
  ```

---

## ⑥ 🛠️ Production Readiness & Architecture Review Checklist

- [ ] **Adopt an `85% / 15%` Split-Plane Architecture**: Index unstructured documents (`SharePoint / Confluence / Slack`) in a central hybrid `BM25 + Vector` store while querying volatile transactional records (`Workday / Salesforce / Live Jira`) via targeted on-demand federation.
- [ ] **Gate Live Federated Calls Behind a `10ms` Source-Intent Classifier**: Never fan out every user query to all 8 SaaS REST APIs unconditionally (`0.0%` `HTTP 429` throttling).
- [ ] **Maintain a Pre-Flattened Cross-SaaS Identity Graph (`Roaring Bitmap ACLs`)**: Synchronize `Entra ID`, `Okta`, and SaaS-local permissions into bitmap filters evaluated in `2ms` at query time.
- [ ] **Normalize Heterogeneous Plane-A and Plane-B Results with a `28ms` `Cross-Encoder` Reranker**: Never interleave raw uncalibrated scores from different SaaS search APIs.
- [ ] **Use Event-Driven `Webhook / Graph Delta` Connectors Instead of Full-Corpus Polling**: Keep unstructured document and ACL updates under 15 minutes while preserving SaaS API rate-limit budgets.
