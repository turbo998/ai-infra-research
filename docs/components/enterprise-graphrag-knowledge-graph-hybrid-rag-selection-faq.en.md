# Enterprise `GraphRAG` vs. Hybrid Vector/BM25 RAG Architecture & TCO Inflection-Point Selection Guide: When Is `GraphRAG` Worth It, and How Do `LazyGraphRAG / LightRAG` + Metadata Graphs Cut Indexing Bills by `97.4%`?

> **Language / 语言**: **English (Current)** | [中文](./enterprise-graphrag-knowledge-graph-hybrid-rag-selection-faq.md)  
> **Scope**: End-to-end production reference architecture and TCO decision guide for Cloud & AI Solution Architects, Enterprise RAG Leads, and Knowledge Platform Engineers.  
> **Provenance**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`). Deterministic evaluation model sealed in [`experiments/enterprise_graphrag_knowledge_graph_hybrid_rag_selection/results.json`](../../experiments/enterprise_graphrag_knowledge_graph_hybrid_rag_selection/results.json).

![Enterprise GraphRAG vs. Hybrid Vector RAG & LazyGraphRAG Summary](../../assets/svg/enterprise-graphrag-knowledge-graph-hybrid-rag-selection-faq-summary.svg)

---

## ① 📌 Executive Summary (3-Minute Architect Verdict)

Every week, enterprise customers ask Cloud & AI Solution Architects: *"Should we replace our vector RAG system with `GraphRAG` across our `1,000,000` internal documents (`20,000,000` chunks)?"* Teams that blindly run full-corpus frontier-LLM `GraphRAG` hit two painful production walls—while teams that stick strictly to flat vector search fail on multi-hop reasoning:
1. **The `$148,000` Upfront `GraphRAG` Indexing Bill Shock (Tier 3 Full-Corpus Frontier `GraphRAG`, `$215,000/month` TCO)**: Running a frontier LLM on all 20M chunks to extract entities, relationships, gleanings, and hierarchical Leiden community summaries burns **`$148,000` before a single user asks a question**—and when 15,000 documents change daily, static pre-computed community reports become stale and impossible to prune cleanly for GDPR deletions;
2. **Overkill & Latency on the `80%` Single-Hop Lookup Workload (`4,850ms` P95 Latency in Tier 3)**: For the `80%` of enterprise queries that ask for a specific clause or fact (*"What is the SLA credit in Contract #8821?"*), standard **Hybrid `BM25 + Dense Vector + Reranker` (Tier 2)** is **7x faster (`690ms`)** and more accurate (`95.8%` vs `93.2%`);
3. **Where Flat Vector RAG Collapses (`34.0%–58.5%` Accuracy in Tiers 1–2)**: On the **`20%` of queries requiring 2–3 hop relational bridging** (*"Which Tier-1 products depend on sub-tier chemical suppliers linked to the port strike?"*) or corpus-wide thematic synthesis, top-K vector search fails because the connecting facts live in separate documents with zero lexical overlap!

Evaluated across a **1,000,000-document (`20M` chunk) enterprise corpus (`80%` single-hop lookups + `20%` multi-hop/global queries)**, the four architectures compare as follows:

| Architectural Tier | Core Retrieval & Graph Stack | Single-Hop QA (`80%` Load) | Multi-Hop / Global QA (`20%` Load) | 1M-Doc Upfront Index Cost | Query P95 Latency | Monthly Index + Serve TCO | Production Verdict |
|---|---|---|---|---|---|---|---|
| **🟥 Tier 1: Naive Dense Vector RAG Only** | Top-5 cosine vector similarity only | `81.5%` | `34.0%` ✕ *(blind to multi-hop)* | `$450` | `580 ms` | `$88,000 / mo` ✕ *(incl. rework)* | ⛔ Fails on both exact SKUs & multi-hop chains |
| **🟨 Tier 2: Standard Hybrid `BM25 + Vector + Rerank`** | Lexical + dense vector + RRF + cross-encoder | `95.8%` 🏆 | `58.5%` ✕ *(breaks at >=2 hops)* | `$1,200` | `690 ms` | `$42,000 / mo` | ✔️ **Best-in-class baseline for 80% single-hop QA** |
| **🟦 Tier 3: 100% Full-Corpus Frontier `GraphRAG`** | Frontier LLM entity extraction + upfront Leiden summaries | `93.2%` *(loses exact clauses)* | `94.6%` | **`$148,000` ✕** *(bill shock)* | `4,850 ms` ✕ *(slow map-reduce)* | `$215,000 / mo` ✕ *(rebuild cost)* | ⛔ 120x indexing cost & brittle on daily doc churn |
| **🟩 Tier 4: Query-Routed Hybrid Vector (`80%`) + `LazyGraphRAG / LightRAG` (`20%`)** | **Deterministic Metadata FKs + `GLiNER / 8B` graph build (`$3,800`) + 80% fast Hybrid Vector + 20% `LazyGraphRAG / LightRAG` 2-hop subgraph** | **`96.8%` 🏆** | **`96.2%` 🏆** *(traverses 3-hop chains)* | **`$3,800` 🏆** *(-97.4% vs Tier 3)* | **`820 ms` 🏆** *(5.9x faster)* | **`$29,500 / mo` 🏆** *(-86.3%)* | 🌟 **Recommended Enterprise Graph + Vector Standard** |

> 💡 **Golden Rule for Enterprise GraphRAG Architects**: **"Never run upfront frontier-LLM entity extraction and Leiden community summarization across 100% of an enterprise corpus! Instead: (1) build 70% of your graph edges for free from structured document/ERP metadata (`vendor_id`, `contract_id`, `citations`) and extract remaining entities with a specialized `GLiNER / Distilled 8B` model (`$3,800` vs `$148,000` per 1M docs); (2) route `80%` single-hop lookup queries to fast `BM25 + Vector + Reranker` (`~550ms`); and (3) route only the `20%` multi-hop and global-theme queries to `LazyGraphRAG / LightRAG` on-demand subgraph traversal."**

---

## ② 🎯 Customer & Architect Deep-Dive Q&A (3 GraphRAG Cost, Routing & Incremental Update Pain Points)

### Q1: How do we cut the upfront indexing cost of 1,000,000 documents from `$148,000` down to `$3,800` (`-97.4%`) without losing multi-hop accuracy (`96.2%`)?
- **Eliminating Upfront Frontier-LLM Extraction & Pre-Computed Community Reports**:
  1. **Metadata Foreign Keys + `GLiNER / 8B` Extraction**: Enterprise documents already contain structured relational metadata (author, customer ID, part numbers, regulatory citations). Combining deterministic metadata edges with a local **`GLiNER` / `8B FP8` extractor** avoids calling a frontier LLM on 20 million chunks.
  2. **`LazyGraphRAG` On-Demand Synthesis**: Instead of pre-summarizing thousands of hierarchical Leiden communities upfront (95% of which are never queried), `LazyGraphRAG / LightRAG` stores only the lightweight entity-relation graph and synthesizes community summaries **lazily at query time** for the specific subgraph touched by the user's question.

### Q2: How do we handle `15,000` daily document updates and GDPR deletions without re-clustering the entire graph?
- **Edge-Level Provenance (`source_doc_id`) + ACL Inheritance**:
  - Every graph edge stores `source_doc_ids[]` and inherits the ACL security groups of its source documents.
  - Adding a document appends edges in `< 2s`; deleting a document decrements `source_doc_id` references and prunes orphan edges in `< 1s` without rebuilding global summaries.

---

## ③ 🧭 Visual Architecture Diagrams (Decision Tree + Hybrid Vector/LazyGraphRAG Topology + Sequence Flow)

### 3.1 Enterprise GraphRAG vs. Hybrid Vector RAG Selection Decision Tree

```mermaid
flowchart TD
    A["Enterprise Knowledge Retrieval Selection<br/>(1,000,000 Documents / 20,000,000 Chunks)"] --> B{"What is the query's reasoning topology?"}
    B -- "Single-hop fact / clause / manual lookup (80% of traffic)" --> C["🟨 Standard Hybrid RAG (BM25 + Vector + Reranker)<br/>(95.8% Accuracy | ~550ms Latency | No Graph Overhead)"]
    B -- "2–3 hop relational chain or corpus-wide theme synthesis (20%)" --> D{"How is the knowledge graph constructed and queried?"}
    D -- "Upfront frontier-LLM extraction + pre-computed Leiden summaries (Tier 3)" --> E["⛔ $148K Indexing Bill Shock & 4,850ms Latency!<br/>Cannot handle daily incremental updates or instant GDPR deletes"]
    D -- "Metadata FKs + GLiNER/8B + LazyGraphRAG/LightRAG On-Demand (Tier 4)" --> F["🟩 Tier 4: 80% Hybrid Vector + 20% LazyGraphRAG/LightRAG Subgraph 🏆<br/>(96.8% Single-Hop / 96.2% Multi-Hop | $3,800 Index (-97.4%) | 820ms P95)"]
    style C fill:#1e3a8a,stroke:#3b82f6,stroke-width:2px,color:#f8fafc
    style F fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#f8fafc
    style E fill:#7f1d1d,stroke:#ef4444,color:#fecaca
```

### 3.2 Tier 4 Low-Cost Graph Indexing + 80/20 Query-Routed `Hybrid Vector + LazyGraphRAG` Reference Architecture

```mermaid
flowchart LR
    subgraph Ingest["1. Low-Cost Incremental Graph Builder ($3,800 / 1M Docs)"]
        Docs["📄 1M Enterprise Docs"] --> Extract["⚡ Metadata FK Extractor + GLiNER / 8B NER<br/>(Zero frontier LLM indexing bill | -97.4% cost)"]
        Extract --> Store["🗄️ Unified Store (Postgres pgvector + Apache AGE / Neo4j)<br/>Stores Vector Chunks + ACL-Tagged Edges (with source_doc_id)"]
    end

    subgraph Router["2. Query Complexity Router"]
        Query["User Query"] --> Cls["🧭 Intent Classifier (<5ms)<br/>Single-Hop Lookup vs Multi-Hop / Global"]
    end

    subgraph Exec["3. Dual-Path Retrieval Execution"]
        Cls -->|80% Single-Hop Lookup| Fast["🚀 Path A: BM25 + Vector + Reranker<br/>(~550ms verbatim clause retrieval)"]
        Cls -->|20% Multi-Hop / Global Theme| Graph["🕸️ Path B: LightRAG 2-Hop Subgraph + LazyGraphRAG<br/>ACL-filtered PPR subgraph traversal + on-demand synthesis"]
        Store --> Fast & Graph
    end
```

### 3.3 End-to-End Sequence: Resolving a 3-Hop Supply-Chain Contagion Query (`Product A -> Part B -> Supplier C -> Port D`)

```mermaid
sequenceDiagram
    autonumber
    participant User as Supply-Chain Risk Director
    participant Router as Query Complexity Router
    participant Graph as Hybrid Graph + Vector Engine (LightRAG / Apache AGE)
    participant LLM as Synthesis LLM

    User->>Router: Ask: "Which Tier-1 products are exposed to the Rotterdam port strike?" (spans 3 separate docs)
    Router->>Router: Classify as `MULTI_HOP_RELATIONAL` -> route to Path B (`LazyGraphRAG / LightRAG`)
    Router->>Graph: 1. Anchor seed entity `Port: Rotterdam`
    Graph->>Graph: 2. Traverse ACL-filtered 3-hop subgraph with Personalized PageRank (PPR) pruning:<br/>`Rotterdam <--ships_via-- ChemCorp C <--supplies-- Cell B <--installed_in-- Product A`
    Graph-->>LLM: 3. Retrieve the 4 exact source contract chunks backing the 3-hop path
    LLM-->>User: Return complete 3-hop contagion chain with verbatim contract citations in 820ms (96.2% accuracy vs 34% vector-only)!
```

---

## ④ 📊 4-Tier Architecture Decision Matrix

| Evaluation Dimension | 🟥 Tier 1: Naive Vector Only | 🟨 Tier 2: Standard Hybrid `BM25+Vector` | 🟦 Tier 3: Full Frontier `GraphRAG` | 🟩 Tier 4: Routed Hybrid + `LazyGraphRAG / LightRAG` |
|---|---|---|---|---|
| **Single-Hop QA Accuracy (`80%` Load)** | `81.5%` | `95.8%` | `93.2%` | 🏆 **`96.8%` (routed to fast BM25+Vector+Rerank)** |
| **Multi-Hop / Global QA (`20%` Load)** | `34.0%` ✕ | `58.5%` ✕ | `94.6%` | 🏆 **`96.2%` (2–3 hop PPR subgraph + LazyGraphRAG)** |
| **1M-Doc Upfront Indexing Cost** | `$450` | `$1,200` | `$148,000` ✕ (token bill shock) | 🏆 **`$3,800` (metadata FKs + GLiNER/8B, -97.4%)** |
| **Blended Query P95 Latency** | `580 ms` | `690 ms` | `4,850 ms` ✕ | 🏆 **`820 ms` (5.9x faster than full GraphRAG)** |
| **Monthly Index + Serving TCO** | `$88,000 / mo` (incl. rework) | `$42,000 / mo` | `$215,000 / mo` | **`$29,500 / mo` 🏆 (-86.3%)** |

---

## ⑤ 🧪 Quantitative Case Study, Interactive Sizer & 7-Card Visual Report

- **Interactive HTML Sizer**: 👉 [Open Enterprise GraphRAG vs. Hybrid Vector RAG & LazyGraphRAG Sizer](../reports/enterprise-graphrag-knowledge-graph-hybrid-rag-sizer.html)
- **7-Card 4:5 Visual Benchmark Report**: 👉 [Open 7-Card Visual Benchmark Deck](../reports/enterprise-graphrag-knowledge-graph-hybrid-rag-selection-cards.html)
- **Deterministic Verification**:
  ```bash
  python experiments/enterprise_graphrag_knowledge_graph_hybrid_rag_selection/run.py --check
  python -m unittest tests/test_enterprise_graphrag_knowledge_graph_hybrid_rag_selection.py
  ```

---

## ⑥ 🛠️ Production Readiness & Architecture Review Checklist

- [ ] **Audit Query Topology Before Adopting `GraphRAG`**: Keep single-hop factual and clause-lookup workloads (`80%` of typical enterprise queries) on fast `BM25 + Vector + Reranker` hybrid search.
- [ ] **Never Run Upfront Frontier-LLM Entity Extraction Across Millions of Chunks**: Construct graph edges from structured document/ERP metadata plus a lightweight `GLiNER / 8B` entity extractor (`-97.4%` indexing cost).
- [ ] **Use `LazyGraphRAG / LightRAG` On-Demand Community Synthesis Instead of Pre-Computing Static Leiden Community Reports**: Enable `< 2s` incremental document ingestion and `< 1s` GDPR deletion.
- [ ] **Attach `source_doc_id` and Per-Document `ACL` Security Tags to Every Graph Edge**: Enforce user ACL filters at every hop of graph traversal so multi-hop reasoning never leaks restricted relationships.
