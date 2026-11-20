# Enterprise `Agentic RAG` vs. Single-Hop Hybrid RAG Architecture & Cost/Latency Inflection Selection Guide: When Is Standard 1-Hop RAG Enough, When Do You Need Multi-Step `Agentic RAG`, and How Do You Prevent `9.4s` ReAct Loops on Simple FAQs?

> **Language / 语言**: **English (Current)** | [中文](./enterprise-agentic-rag-vs-standard-rag-router-query-decomposition-selection-faq.md)  
> **Scope**: End-to-end production reference architecture and TCO selection guide for Cloud & AI Solution Architects, Knowledge Platform Leads, and Applied AI Engineers deciding between Single-Hop Hybrid RAG and Multi-Step Agentic RAG (`Query Decomposition + Parallel Sub-Query DAG + Corrective RAG [CRAG]`).  
> **Provenance**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`). Deterministic evaluation model sealed in [`experiments/enterprise_agentic_rag_vs_standard_rag_router_query_decomposition_selection/results.json`](../../experiments/enterprise_agentic_rag_vs_standard_rag_router_query_decomposition_selection/results.json).

![Enterprise Agentic RAG vs. Single-Hop Hybrid RAG Routing & Sizing Summary](../../assets/svg/enterprise-agentic-rag-vs-standard-rag-router-query-decomposition-selection-faq-summary.svg)

---

## ① 📌 Executive Summary (3-Minute Architect Verdict)

Every enterprise knowledge team today faces the same architectural dilemma: **"Should we stick with fast Single-Hop Hybrid RAG (`BM25 + Vector + Reranker`), or upgrade everything to multi-step `Agentic RAG` (`Plan -> Search -> Reflect -> Rewrite -> Answer`)?"** Treating this as an all-or-nothing choice creates two opposite production disasters:
1. **Trap #1: 100% Single-Hop Vector RAG Only (Tier 1, `44.0%` Accuracy on Multi-Hop / Comparative Queries)**: When an executive asks *"Compare Q3 EMEA cloud gross margin against Q2 APAC gross margin and explain which new enterprise contract clauses drove the variance"*, a single embedding vector cannot simultaneously retrieve three separate targets (`Q3 SQL metric`, `Q2 SQL metric`, and `Contract PDF clauses`)—retrieving only Q3 chunks and hallucinating the rest!
2. **Trap #2: Routing 100% of Traffic into an Unbounded `ReAct Agentic RAG` Loop (Tier 2, `9,450ms` P95 Latency, `4.8` LLM Calls/Query, `$318,000/month` for 30M Queries)**: **75% of enterprise queries are simple single-hop fact lookups** (*"What is the VPN helpdesk number?"* or *"What is the hotel per-diem cap in Tokyo?"*). Forcing those simple lookups through a 4.8-step autonomous reflection loop makes users wait nearly 10 seconds and inflates token bills by **6.6x**!
3. **Trap #3: Infinite Query-Rewrite Loops on Missing Documents**: Without a deterministic **Corrective RAG (`CRAG`) relevance gate** and a hard `max_steps = 2` ceiling, an unconstrained agent rewrites its search query 6–8 times in a row whenever a user asks about a document that hasn't been uploaded yet.

Evaluated across **30M monthly enterprise queries (`75%` single-hop fact lookups + `25%` multi-hop comparative/cross-source queries)**, the four RAG architectures compare as follows:

| Architectural Tier | Retrieval & Agentic Orchestration Stack | Multi-Hop Query Accuracy | Blended Accuracy | Blended `P95` Latency | Avg LLM Calls / Query | 30M-Query Monthly TCO | Production Verdict |
|---|---|---|---|---|---|---|---|
| **🟥 Tier 1: 100% Single-Hop Vector RAG Only** | Embeds raw query into 1 vector search (`Top-5`) | `44.0%` ✕ *(fails comparisons)* | `68.5%` ✕ | `680 ms` | `1.00x` | `$48,000 / mo` | ⛔ Fails multi-entity comparisons & SQL+Doc queries |
| **🟨 Tier 2: 100% Unbounded `ReAct Agentic RAG` (`3–6 Hops`)** | Sends every query into a serial ReAct reflection loop | `91.5%` | `92.8%` | **`9,450 ms` ✕** *(10s freeze)* | **`4.80x` ✕** | **`$318,000 / mo` ✕** *(6.6x cost blowup)* | ⛔ Wastes 5x tokens & 9s latency on 75% simple FAQs |
| **🟦 Tier 3: Static 2-Hop Rewrite + Serial Sub-Queries** | Always rewrites with LLM (+450ms) & runs hops serially | `79.0%` | `85.4%` | `3,280 ms` ✕ | `2.40x` | `$146,000 / mo` | ⚠️ Serial sub-query execution is too slow |
| **🟩 Tier 4: `12ms` Complexity Router (`75%` 1-Hop Fast Path + `25%` Parallel `DAG Agentic RAG` + `CRAG` Gate)** | **`12ms` classifier routes 75% to `520ms` 1-Hop Hybrid+TEI, and 25% to Parallel Sub-Query Fan-Out (`max_steps=2`)** | **`94.6%` 🏆** *(parallel recall)* | **`95.8%` 🏆** | **`1,120 ms` 🏆** *(-88.1% vs Tier 2)* | **`1.35x` 🏆** *(on-demand only)* | **`$64,000 / mo` 🏆** *(-79.9% vs Tier 2)* | 🌟 **Recommended Fast/Slow Dual-Track RAG Standard** |

> 💡 **Golden Rule for Enterprise RAG Architects**: **"Never force 100% of user queries through a slow serial ReAct Agentic RAG loop! Place a `12ms` Query-Complexity Classifier at the ingress: route the `75%` single-entity fact lookups directly to a `520ms` Single-Hop Hybrid + `TEI` Reranker fast path (`1.0` LLM call), and route only the `25%` multi-entity/comparative/cross-system queries to a Bounded Parallel Sub-Query Decomposition DAG (`asyncio.gather` across Vector + Semantic SQL + APIs) governed by a `CRAG` relevance gate (`max_steps <= 2`)."**

---

## ② 🎯 Customer & Architect Deep-Dive Q&A (3 Single-Vector Blindness, Serial ReAct Latency & Missing-Doc Loop Pain Points)

### Q1: Why does Single-Hop Vector RAG fail (`44.0%` accuracy) on comparative questions like *"Compare Product A vs. Product B failure rates and supplier penalty clauses"* even if we increase `Top-K` to 20?
- **Single-Vector Centroid Skew & Cross-Source Blindness**:
  - Encoding a compound prompt (*"Product A + Product B + failure rate table + supplier contract"*) into a single embedding vector produces a centroid that is dominated by whichever entity has denser text chunks in the index—often returning 20 chunks about Product A and 0 chunks about Product B!
  - Furthermore, failure rates live in a structured **SQL Data Warehouse**, while penalty clauses live in unstructured **PDF Vector Indexes**.
  - Tier 4's **Parallel Sub-Query Decomposition DAG** breaks the compound prompt into **3 independent sub-queries (`[SQL_Product_A_B, Vec_Clause_A, Vec_Clause_B]`)** in one pass and executes all three **concurrently via `asyncio.gather`**—lifting multi-hop accuracy from `44.0%` to **`94.6%`**!

### Q2: How do we cut `Agentic RAG` latency from `9,450ms` down to `1,120ms` blended P95 (`1,850ms` on the complex path)?
- **Two Architectural Switches: Ingress Complexity Gating (`75/25`) + Parallel DAG Execution**:
  1. **Bypass Agent Loops on 75% of Traffic**: A `12ms` `DeBERTa-v3 / Qwen-1.5B INT8` classifier detects single-entity lookups and routes them straight to the `520ms` 1-Hop Hybrid+TEI fast path;
  2. **Replace Serial `ReAct` with a Parallel Sub-Query DAG**: Instead of `Thought 1 -> Search A (1.5s) -> Thought 2 -> Search B (1.5s) -> Thought 3 -> Search C (1.5s)`, the planner emits all independent sub-queries upfront so they run in parallel (`max(420ms, 380ms, 450ms) = 450ms`).

### Q3: How do we stop an Agentic RAG workflow from rewriting its search query 6–8 times when the answer simply does not exist in the internal knowledge base?
- **Deterministic `CRAG (Corrective RAG)` Reranker Threshold (`0.35`) + Hard `max_steps = 2` Cap**:
  - After Step 1 retrieval, check the top **`TEI Cross-Encoder Reranker` probability score (`max_score ∈ [0, 1]`)**:
    - If `max_score >= 0.65`: Immediately synthesize the final answer (`0` retries);
    - If `0.35 <= max_score < 0.65`: Trigger **at most 1 synonym/broadened query rewrite (`max_steps = 2`)**;
    - If Step 2 still scores `< 0.35`: **Immediately halt and cleanly abstain** (*"No approved internal document covers X"*)—preventing 6-step token burn!

---

## ③ 🧭 Visual Architecture Diagrams (Decision Tree + Fast/Slow Dual-Track DAG Topology + Parallel Multi-Hop Sequence)

### 3.1 Enterprise Single-Hop Hybrid RAG vs. Agentic RAG Selection Decision Tree

```mermaid
flowchart TD
    A["Enterprise Knowledge Retrieval Architecture Selection<br/>(30M Queries/Month | 75% Simple Lookups + 25% Multi-Hop/Cross-Source)"] --> B{"How are incoming user queries routed?"}
    B -- "100% Single-Hop Vector RAG Only (Tier 1)" --> C["⛔ 44.0% Multi-Hop Accuracy!<br/>Single vector misses second entity & cannot query SQL metrics"]
    B -- "100% Unbounded Serial ReAct Agentic RAG (Tier 2)" --> D["⛔ 9,450ms P95 Latency & $318K/Mo Bill!<br/>Wastes 4.8 LLM calls/query on simple helpdesk lookups"]
    B -- "12ms Ingress Query-Complexity Classifier (Fast/Slow Dual-Track)" --> E{"What is the classified query complexity?"}
    E -- "75% Single-Entity / Direct Fact Lookup (Single-Hop)" --> F["🟦 Fast Path: 1-Hop Hybrid BM25 + Vector + L4 TEI Rerank<br/>(520ms P95 Latency | 1.0 LLM Call)"]
    E -- "25% Multi-Entity Comparison / Multi-Hop / Doc+SQL (Complex)" --> G["🟩 Agentic Path: Parallel Sub-Query Decomposition DAG + CRAG Gate 🏆<br/>(94.6% Multi-Hop Accuracy | max_steps=2 Cap | -79.9% Monthly TCO)"]
    style F fill:#1e3a8a,stroke:#3b82f6,stroke-width:2px,color:#f8fafc
    style G fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#f8fafc
    style C fill:#7f1d1d,stroke:#ef4444,color:#fecaca
    style D fill:#7f1d1d,stroke:#ef4444,color:#fecaca
```

### 3.2 Tier 4 `12ms` Complexity Router + Parallel Sub-Query DAG + `CRAG` Relevance Gate Reference Architecture

```mermaid
flowchart LR
    subgraph Ingress["1. Ingress Complexity Router (12ms)"]
        UserQ["❓ 30M Queries/Mo"] --> Router["🧭 INT8 Query-Complexity Classifier<br/>Detects Single-Hop vs. Multi-Hop/Federated"]
    end

    subgraph Fast["2. Fast Track: 75% Single-Hop Hybrid RAG (520ms)"]
        Router -->|75% Direct Lookups| OneHop["⚡ Hybrid BM25 + 512d Vector + L4 TEI Rerank<br/>Single-Pass Synthesis (1.0x LLM Call)"]
    end

    subgraph Agentic["3. Agentic Track: 25% Bounded Parallel DAG (max_steps = 2)"]
        Router -->|25% Comparative / Multi-Hop| Plan["🧩 Sub-Query Decomposer<br/>Emits Parallel Retrieval DAG"]
        Plan --> Gather["🚀 asyncio.gather Federated Fan-Out<br/>Parallel: Vector(Entity A) + Vector(Entity B) + Semantic SQL"]
        Gather --> Gate["🛡️ CRAG Reranker Gate (Score >= 0.35)<br/>Synthesize if >=0.65; Max 1 Rewrite if <0.35 else Abstain"]
    end
```

### 3.3 End-to-End Sequence: Answering a Cross-Region SQL + Contract Comparison Query in `1.4s` via Parallel DAG Fan-Out

```mermaid
sequenceDiagram
    autonumber
    participant User as Enterprise Analyst
    participant Cls as 12ms Complexity Router
    participant Plan as Sub-Query Decomposer (8B FP8)
    participant Fed as Federated Retrieval (Vector + Semantic SQL)
    participant CRAG as L4 TEI Reranker & CRAG Gate
    participant LLM as Synthesis LLM (70B / GPT-4o)

    User->>Cls: 1. Query: "Compare Q3 EMEA vs Q2 APAC cloud gross margin & cite new contract SLA clauses"
    Cls->>Plan: 2. Classify as MULTI_HOP_FEDERATED in 12ms -> route to Bounded Agentic DAG
    Plan-->>Fed: 3. Decompose into 3 independent targets in 180ms: [SQL_Q3_EMEA, SQL_Q2_APAC, Vec_SLA_Clauses]
    par Execute all 3 sub-queries concurrently in asyncio.gather (Total elapsed = 380ms!)
        Fed->>Fed: Sub-Query 1: Query Semantic SQL Layer for Q3 EMEA Margin
        Fed->>Fed: Sub-Query 2: Query Semantic SQL Layer for Q2 APAC Margin
        Fed->>Fed: Sub-Query 3: Hybrid Vector+BM25 Search for New Contract SLA Clauses
    end
    Fed->>CRAG: 4. Score contract chunks on L4 TEI Reranker in 34ms (max_score = 0.88 >= 0.35 CRAG pass!)
    CRAG->>LLM: 5. Pass verified Q3/Q2 SQL tables + Top-5 reranked contract clauses to Synthesis LLM
    LLM-->>User: 6. Stream cited comparative report in 1.4s total (vs 9.45s in serial ReAct)!
```

---

## ④ 📊 4-Tier Architecture Decision Matrix

| Evaluation Dimension | 🟥 Tier 1: 100% 1-Hop Vector RAG | 🟨 Tier 2: 100% Unbounded `ReAct` | 🟦 Tier 3: Static 2-Hop Serial | 🟩 Tier 4: `75/25` Router + Parallel `DAG` + `CRAG` |
|---|---|---|---|---|
| **Multi-Hop Query Accuracy** | `44.0%` ✕ (single-vector miss) | `91.5%` | `79.0%` | 🏆 **`94.6%` (parallel sub-query DAG + SQL federation)** |
| **Blended Overall Accuracy** | `68.5%` ✕ | `92.8%` | `85.4%` | 🏆 **`95.8%` (fast 1-hop + deep multi-hop + CRAG)** |
| **Blended `P95` Latency** | `680 ms` | `9,450 ms` ✕ (4.8 serial hops) | `3,280 ms` ✕ | 🏆 **`1,120 ms` (`520ms` on 75% fast path; parallel DAG)** |
| **Avg LLM Calls per Query** | `1.00x` | `4.80x` ✕ (token explosion) | `2.40x` | 🏆 **`1.35x` (decomposes only the 25% complex share)** |
| **30M-Query Monthly TCO** | `$48,000 / mo` (broken multi-hop) | `$318,000 / mo` ✕ | `$146,000 / mo` | **`$64,000 / mo` 🏆 (-79.9% vs Tier 2)** |

---

## ⑤ 🧪 Quantitative Case Study, Interactive Sizer & 7-Card Visual Report

- **Interactive HTML Sizer**: 👉 [Open Enterprise Agentic RAG vs. Single-Hop Hybrid RAG Sizer](../reports/enterprise-agentic-rag-vs-standard-rag-router-query-decomposition-sizer.html)
- **7-Card 4:5 Visual Benchmark Report**: 👉 [Open 7-Card Visual Benchmark Deck](../reports/enterprise-agentic-rag-vs-standard-rag-router-query-decomposition-selection-cards.html)
- **Deterministic Verification**:
  ```bash
  python experiments/enterprise_agentic_rag_vs_standard_rag_router_query_decomposition_selection/run.py --check
  python -m unittest tests/test_enterprise_agentic_rag_vs_standard_rag_router_query_decomposition_selection.py
  ```

---

## ⑥ 🛠️ Production Readiness & Architecture Review Checklist

- [ ] **Place a `< 15ms` Query-Complexity Classifier in Front of Every Enterprise RAG Pipeline**: Keep `70%–80%` of single-entity lookups on the `520ms` 1-Hop Hybrid+TEI fast path.
- [ ] **Execute Decomposed Sub-Queries in Parallel (`asyncio.gather` DAG) Rather Than Serial `ReAct` Loops**: Cut multi-entity comparison retrieval latency by `3.5x`.
- [ ] **Enforce a Deterministic `CRAG` Reranker Relevance Gate (`score >= 0.35`) and Hard `max_steps = 2` Ceiling**: Allow at most one query rewrite when initial retrieval scores are low, then cleanly abstain to prevent infinite search loops.
- [ ] **Federate Unstructured Document Indexes with Structured `Semantic SQL / REST APIs`**: Route quantitative metric sub-queries to governed SQL/API tools while routing clause/policy sub-queries to the vector database.
- [ ] **Deduplicate Retrieved Chunks by `chunk_id` Across Parallel Sub-Queries Before Synthesis**: Prevent overlapping sub-queries from stuffing duplicate passages into the final LLM prompt.
