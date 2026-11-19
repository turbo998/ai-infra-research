# Enterprise `Text-to-SQL`, Semantic Metric Layer (`dbt / Cube / Fabric IQ / Snowflake Cortex`) & Governed Data Analytics Agent Selection Guide: Raw Physical `DDL` Dump vs. Schema `RAG` vs. Fine-Tuned SQL Models vs. Governed Semantic Layer + Cell-Value Entity Index + `SQLGlot` AST Cost/RLS Gate

> **Language / 语言**: **English (Current)** | [中文](./enterprise-text2sql-semantic-layer-governed-bi-agent-selection-faq.md)  
> **Scope**: End-to-end production reference architecture and accuracy/governance guide for Cloud & AI Solution Architects, Chief Data Officers (`CDO`), BI Platform Leads, and Data Warehouse Architects.  
> **Provenance**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`). Deterministic evaluation model sealed in [`experiments/enterprise_text2sql_semantic_layer_governed_bi_agent_selection/results.json`](../../experiments/enterprise_text2sql_semantic_layer_governed_bi_agent_selection/results.json).

![Enterprise Text-to-SQL, Semantic Layer & Governed BI Agent Selection Summary](../../assets/svg/enterprise-text2sql-semantic-layer-governed-bi-agent-selection-faq-summary.svg)

---

## ① 📌 Executive Summary (3-Minute Architect Verdict)

Every enterprise executive wants a **"Chat-with-Your-Data" Analytics Copilot**—asking *"What was Q3 EMEA Net ARR growth excluding trial churn?"* and getting an instant, trustworthy answer. Yet when an enterprise data warehouse (`Snowflake`, `Microsoft Fabric`, `Databricks`, `BigQuery`, `PostgreSQL`) spans **`650` physical tables, `50 Billion` fact-table rows, `180` corporate KPIs, and `1,800,000` monthly natural-language queries**, **raw `Text-to-SQL` against physical tables fails catastrophically in production**:
1. **Failure #1: Conflicting KPI Numbers Across Executives (`38.0%–74.0%` Metric Consistency, `44.5%` Raw DDL Accuracy, Tiers 1–3)**: Computing a certified corporate metric like `Net ARR` or `Gross Margin` requires an 8-table join with specific exclusions for test accounts, inter-company transfers, VAT, and credit memos. Asking an LLM to guess those 8-way physical joins on the fly produces three different answers for the same question!
2. **Failure #2: Silent `0-Row` Empty Results on Dirty Categorical Strings (Cell-Value Miss)**: The user asks for *"McDonald's sales in the UK"*; the LLM writes syntactically valid SQL (`WHERE merchant = 'McDonalds' AND country = 'UK'`), which returns `0 rows` because the database actually stores `'McDonald''s Corp'` and `'GBR'`!
3. **Failure #3: Runaway `$5,000` Full-Table Scans & Shared-Admin RLS Bypasses (`28.5%` Scan Risk in Tier 1)**: One hallucinated join missing a date-partition predicate scans 50 billion rows and blows the monthly warehouse budget.

Evaluated across a **650-table data warehouse (`50B` fact rows, `180` governed metrics, `1.8M` monthly queries)**, the four data agent architectures compare as follows:

| Architectural Tier | Core Analytics Agent Stack | Business KPI Accuracy | Cross-User KPI Consistency | Runaway Full-Scan Risk | `RLS` & Masking Compliance | Monthly Warehouse + BI TCO | Production Verdict |
|---|---|---|---|---|---|---|---|
| **🟥 Tier 1: Raw Physical `DDL` Dump (`650 Tables`)** | Dumps `CREATE TABLE` DDL into prompt -> raw physical SQL | `44.5%` ✕ *(fails 8-way joins)* | `38.0%` ✕ *(invents KPI formulas)* | `28.5%` ✕ *(misses partition filters)* | `52.0%` ✕ *(shared admin account)* | `$174,000 / mo` ✕ *(runaway scans)* | ⛔ Works only on 5-table toy demos |
| **🟨 Tier 2: Vector Schema `RAG` + Few-Shot SQL** | Retrieves top-15 physical tables + golden SQL examples | `76.2%` | `68.5%` ✕ | `14.2%` ✕ | `78.0%` | `$118,000 / mo` ✕ | ⚠️ Still guesses physical joins & misses cell values |
| **🟦 Tier 3: Fine-Tuned SQL LLM (`SQLCoder / 70B LoRA`)** | Fine-tuned on historical SQL against physical tables | `81.8%` | `74.0%` ✕ *(breaks on schema refactor)* | `9.5%` ✕ | `82.0%` | `$96,000 / mo` | ⚠️ Brittle whenever data engineers refactor tables |
| **🟩 Tier 4: Governed Semantic Layer + Value Index + `SQLGlot` AST Gate** | **LLM queries Semantic Metrics (`dbt / Cube / Fabric IQ / Cortex`) + Cell-Value Entity Index + `SQLGlot` AST/`EXPLAIN` Gate + OBO `RLS`** | **`97.4%` 🏆** *(+52.9 pts vs Tier 1)* | **`100.0%` 🏆** *(single source of truth)* | **`0.0%` 🏆** *(10GB EXPLAIN cap)* | **`100.0%` 🏆** *(OBO user role)* | **`$33,000 / mo` 🏆** *(-81.0%)* | 🌟 **Recommended Enterprise BI Agent Standard** |

> 💡 **Golden Rule for Enterprise Data & AI Solution Architects**: **"Replace raw `Text-to-Physical-SQL` with `Text-to-Semantic-Layer`! Define corporate KPIs, join graphs, and fiscal calendars once inside a Governed Semantic Layer (`Microsoft Fabric IQ / Power BI Semantic Model`, `Databricks Metric Views / Genie`, `Snowflake Cortex Analyst`, `dbt Semantic Layer`, or `Cube`). Have the LLM emit structured metric/dimension requests grounded by a Categorical Cell-Value Index (`'UK' -> 'GBR'`), and gate every compiled SQL statement through a deterministic `SQLGlot` AST + Warehouse `EXPLAIN` byte ceiling executed under the user's Delegated `OBO` role."**

---

## ② 🎯 Customer & Architect Deep-Dive Q&A (3 Metric Drift, Empty-Row & Runaway Warehouse Scan Pain Points)

### Q1: Why does targeting a Governed Semantic Metric Layer (`dbt / Cube / Fabric IQ / Cortex`) jump KPI accuracy from `44.5%–76.2%` to `97.4%` with `100.0%` consistency?
- **First-Principles Separation of Intent vs. Physical Compilation**:
  - Instead of forcing an LLM to write a 60-line SQL query joining 8 cryptic physical tables, the LLM selects certified semantic objects (`metrics: ["net_arr"]`, `dimensions: ["customer__region"]`, `time_grain: "fiscal_quarter"`).
  - The **Semantic Layer Compiler** deterministically expands those semantic objects into the exact certified SQL join path, exclusion predicates, and custom `4-5-4` fiscal calendar math—so two executives asking the same question are mathematically guaranteed to get the exact same formula (`100.0%` consistency).

### Q2: How does the Categorical Cell-Value Entity Index eliminate silent `0-row` results caused by string mismatches (`'UK'` vs. `'GBR'`)?
- **Pre-Query Literal Grounding (`MinHash LSH + Vector Index`)**:
  - Distinct string values from high-cardinality dimension columns (`merchant_name`, `country_code`, `product_sku`) are indexed offline.
  - Before the LLM generates a filter predicate, candidate entities from the user's prompt are resolved against the Cell-Value Index (`"UK" -> "GBR"`, `"McDonalds" -> "McDonald's Corp"`), injecting exact database literals into the prompt.

### Q3: How does the `SQLGlot` AST + `EXPLAIN` Cost Gate guarantee `0.0%` runaway warehouse bills and `100.0%` row-level security?
- **Three Deterministic Pre-Execution Guardrails**:
  1. **AST Read-Only & Partition Verification (`SQLGlot`)**: Parses the compiled SQL into an AST, rejecting any non-`SELECT` statement and requiring a partition-key predicate (`event_date >= ...`) on multi-billion-row fact tables;
  2. **Warehouse `EXPLAIN` Byte-Scan Ceiling**: Runs a zero-cost `EXPLAIN` plan and blocks execution if estimated bytes scanned exceed the tenant cap (e.g., `10 GB`);
  3. **Delegated `OBO` Session Execution**: Executes the query using the calling user's identity (`SET ROLE / EXECUTE AS USER`), enforcing native warehouse Row-Level Security (`RLS`) and Dynamic Data Masking.

---

## ③ 🧭 Visual Architecture Diagrams (Decision Tree + Semantic Layer Topology + Sequence Flow)

### 3.1 Enterprise Text-to-SQL & Governed Data Analytics Agent Selection Decision Tree

```mermaid
flowchart TD
    A["Enterprise Conversational BI & Text-to-SQL Selection<br/>(650 Physical Tables + 50B Fact Rows + 180 Corporate KPIs)"] --> B{"Does the LLM write raw multi-table JOINs and KPI formulas<br/>directly against physical warehouse tables?"}
    B -- "Yes (Raw DDL Dump or Physical Schema RAG, Tiers 1–3)" --> C["⛔ KPI Drift (38%–74% Consistency) & Runaway Scan Risk!<br/>Hallucinates join paths & breaks whenever physical tables refactor"]
    B -- "No (Targets a Governed Semantic Metric Layer)" --> D{"Includes Categorical Cell-Value Entity Linking<br/>+ SQLGlot AST/EXPLAIN Cost Gate + Delegated OBO RLS?"}
    D -- "No (Unlinked literals or un-gated execution)" --> E["⚠️ Silent 0-Row String Misses & $5K Full-Table Scans"]
    D -- "Yes (Semantic Layer + Value Index + SQLGlot Gate + OBO RLS)" --> F["🟩 Tier 4: Governed Semantic Layer + Value Index + AST Gate 🏆<br/>(97.4% KPI Accuracy | 100% Consistency | 0.0% Scan Risk | -81.0% TCO)"]
    style F fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#f8fafc
    style C fill:#7f1d1d,stroke:#ef4444,color:#fecaca
```

### 3.2 Tier 4 Governed Semantic Metric Layer + Cell-Value Index + `SQLGlot` AST Gate Reference Architecture

```mermaid
flowchart LR
    subgraph Grounding["1. Metric & Cell-Value Grounding"]
        User["Natural-Language BI Query<br/>(with User OBO Token)"] --> Linker["🔍 Metric & Cell-Value Linker<br/>1. Matches certified metric (net_arr)<br/>2. Resolves 'UK' -> 'GBR' in Value Index"]
    end

    subgraph Semantic["2. Semantic Layer Compilation"]
        Linker --> LLM["🧠 LLM Emits Semantic Query Spec<br/>(Metrics + Dimensions + Fiscal Grain)"]
        LLM --> Compiler["📐 Semantic Metric Compiler<br/>(dbt MetricFlow / Cube / Fabric IQ / Cortex)<br/>Expands certified joins & 4-5-4 calendar"]
    end

    subgraph Gate["3. SQLGlot AST Gate & OBO Execution"]
        Compiler --> AST["🛡️ SQLGlot AST + EXPLAIN Cost Gate<br/>1. Enforces SELECT-only & partition filter<br/>2. Verifies EXPLAIN scan <= 10GB"]
        AST -->|Delegated OBO Role| DW["🏛️ Data Warehouse (Snowflake / Fabric / Databricks)<br/>Enforces Native RLS & Column Masking"]
    end
```

### 3.3 End-to-End Sequence: Answering an Executive `"Last Fiscal Quarter UK Net ARR"` Query with Zero Drift

```mermaid
sequenceDiagram
    autonumber
    participant Exec as Executive / Regional Manager
    participant Agent as Governed BI Agent
    participant ValIdx as Cell-Value Entity Index (LSH + Vector)
    participant Sem as Semantic Layer (dbt / Cube / Fabric IQ)
    participant Gate as SQLGlot AST + EXPLAIN Gate
    participant DW as Data Warehouse (OBO RLS Enabled)

    Exec->>Agent: Ask: "What was UK Net ARR last fiscal quarter?"
    Agent->>ValIdx: Resolve entity `"UK"` against dimension values -> returns `country_iso = 'GBR'`
    Agent->>Sem: LLM emits semantic query: `{metric: "net_arr", filters: ["country_iso='GBR'"], time: "last_fiscal_quarter"}`
    Sem-->>Gate: Compile into certified 8-table SQL with corporate 4-5-4 fiscal calendar bounds
    Gate->>Gate: 1. Verify read-only `SELECT` & mandatory `event_date` partition filter via SQLGlot; 2. Verify `EXPLAIN` scan is 180MB (< 10GB cap)
    Gate->>DW: Execute under caller's Delegated OBO Role (native RLS applied)
    DW-->>Exec: Return 100% finance-certified KPI number + lineage explanation!
```

---

## ④ 📊 4-Tier Architecture Decision Matrix

| Evaluation Dimension | 🟥 Tier 1: Raw `DDL` Dump | 🟨 Tier 2: Physical Schema `RAG` | 🟦 Tier 3: Fine-Tuned Physical SQL LLM | 🟩 Tier 4: Semantic Layer + Value Index + `SQLGlot` |
|---|---|---|---|---|
| **Business KPI Execution Accuracy** | `44.5%` ✕ | `76.2%` | `81.8%` | 🏆 **`97.4%` (certified metric compiler + value linking)** |
| **Cross-Executive KPI Consistency** | `38.0%` ✕ | `68.5%` ✕ | `74.0%` ✕ | 🏆 **`100.0%` (single governed metric definition)** |
| **Runaway Full-Table Scan Risk** | `28.5%` ✕ | `14.2%` ✕ | `9.5%` ✕ | 🏆 **`0.0%` (`SQLGlot` partition check + `10GB EXPLAIN` cap)** |
| **`RLS` & Column Masking Compliance** | `52.0%` ✕ | `78.0%` | `82.0%` | 🏆 **`100.0%` (executes under caller's OBO DB role)** |
| **1.8M-Query Monthly TCO** | `$174,000 / mo` | `$118,000 / mo` | `$96,000 / mo` | **`$33,000 / mo` 🏆 (-81.0%, saves $1.69M/yr)** |

---

## ⑤ 🧪 Quantitative Case Study, Interactive Sizer & 7-Card Visual Report

- **Interactive HTML Sizer**: 👉 [Open Enterprise Text-to-SQL, Semantic Metric Layer & Governed BI Agent Sizer](../reports/enterprise-text2sql-semantic-layer-governed-bi-agent-sizer.html)
- **7-Card 4:5 Visual Benchmark Report**: 👉 [Open 7-Card Visual Benchmark Deck](../reports/enterprise-text2sql-semantic-layer-governed-bi-agent-selection-cards.html)
- **Deterministic Verification**:
  ```bash
  python experiments/enterprise_text2sql_semantic_layer_governed_bi_agent_selection/run.py --check
  python -m unittest tests/test_enterprise_text2sql_semantic_layer_governed_bi_agent_selection.py
  ```

---

## ⑥ 🛠️ Production Readiness & Architecture Review Checklist

- [ ] **Place a Governed Semantic Metric Layer (`Microsoft Fabric IQ / Power BI Semantic Model`, `Databricks Metric Views`, `Snowflake Cortex Analyst`, `dbt Semantic Layer`, or `Cube`) Between the LLM and Physical Warehouse Tables**.
- [ ] **Index High-Cardinality Categorical Column Values (`MinHash LSH + Vector Index`)**: Resolve user shorthand (`"UK"`, `"McDonalds"`) into exact canonical database strings before generating filter predicates.
- [ ] **Encode Custom Corporate Fiscal Calendars (`4-5-4`, Fiscal Year Start) Inside the Semantic Time Dimension**: Never let the LLM guess fiscal quarter date ranges in raw SQL.
- [ ] **Enforce a Deterministic `SQLGlot` AST + Warehouse `EXPLAIN` Gate**: Block non-`SELECT` statements, require partition predicates on large fact tables, and abort queries exceeding a `10 GB` scan threshold.
- [ ] **Execute All Queries Under the Caller's Delegated `OBO` Warehouse Identity**: Enforce native Row-Level Security (`RLS`) and Dynamic Data Masking (`DDM`).
