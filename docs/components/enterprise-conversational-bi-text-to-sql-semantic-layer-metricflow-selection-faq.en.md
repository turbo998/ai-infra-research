# Enterprise Conversational BI & Data Agent Selection Guide: Raw Schema `Text-to-SQL` vs. Governed Semantic Layer (`Cube / dbt MetricFlow / Fabric Semantic Model`), `SQLGlot AST` Guardrails & OAuth Row-Level Security (`RLS`)

> **Language / 语言**: **English (Current)** | [中文](./enterprise-conversational-bi-text-to-sql-semantic-layer-metricflow-selection-faq.md)  
> **Scope**: End-to-end production reference architecture and sizing playbook for Cloud & AI Solution Architects, Chief Data Officers, and Analytics Engineering Leads building "Ask Your Data / ChatBI" agents on `Snowflake`, `Databricks`, `BigQuery`, and `Microsoft Fabric`.  
> **Provenance**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`). Deterministic evaluation model sealed in [`experiments/enterprise_conversational_bi_text_to_sql_semantic_layer_metricflow_selection/results.json`](../../experiments/enterprise_conversational_bi_text_to_sql_semantic_layer_metricflow_selection/results.json).

![Enterprise Conversational BI Text-to-SQL vs Governed Semantic Layer Summary](../../assets/svg/enterprise-conversational-bi-text-to-sql-semantic-layer-metricflow-selection-faq-summary.svg)

---

## ① 📌 Executive Summary (3-Minute Architect Verdict)

When enterprises deploy "Chat with Your Data" agents across **`800+` physical data warehouse tables (`200,000` monthly executive and analyst queries)**, naive **`Raw DDL Text-to-SQL`** pipelines fail catastrophically in production:
1. **Trap #1: 1-to-N Join Fan-Out Inflation & Unstandardized Business Logic (`Tier 1`, `41.5%` Wrong Executive KPI Numbers — Only `58.5%` Accuracy, `$215,000/month` TCO)**:
   - When an executive asks *"Compare Q3 Net ARR and Support Ticket Volume by Region,"* a raw `Text-to-SQL` LLM joins `subscriptions (1:M)` directly with `support_tickets (1:M)` and runs `SUM(arr)`—**multiplying revenue by 12x due to the classic multi-fact Chasm/Fan-Out Trap**! It also forgets mandatory business filters (`WHERE is_test_tenant = false AND status = 'SETTLED'`), producing different numbers from official PowerBI/Tableau dashboards!
2. **Trap #2: Runaway `50TB` Full-Table Scans (`Tier 1`, `8.2%` Runaway Scan Rate)**: A single LLM query missing a date partition filter or containing an accidental Cartesian join burns `$50–$250` of serverless Snowflake/BigQuery compute in seconds!
3. **Trap #3: Shared Service-Account Row-Level Security (`RLS`) Leaks (`12.4%` Unauthorized Exposure Risk)**: Executing LLM-generated SQL over a shared service principal allows a regional manager to query global compensation or competitor territory data!

Evaluated across **800 physical tables and 200,000 monthly natural-language BI queries (`90%` governed KPI analytics, `10%` ad-hoc exploration)**, the four architectures compare as follows:

| Architecture Tier | Metric Governance, SQL Compiler & Security Stack | Executive KPI Accuracy | Runaway Table Scan Rate | `RLS` Exposure Risk | 200K-Query Monthly TCO | Production Verdict |
|---|---|---|---|---|---|---|---|
| **🟥 Tier 1: Raw Physical `DDL` Prompting -> Free-Form SQL** | LLM writes raw `JOIN`s & `SUM`s over physical tables; shared DB user | `58.5%` ✕ *(41.5% wrong KPIs)* | `8.2%` ✕ *(50TB full scans)* | `12.4%` ✕ *(bypasses RLS)* | `$215,000 / mo` ✕ *(scan waste + rework)* | ⛔ Never let an LLM write raw multi-table aggregation SQL in prod |
| **🟨 Tier 2: Vector `Schema-RAG + Few-Shot SQL` (`Vanna` Style)** | Retrieves similar historical SQL queries into prompt | `80.6%` ✕ *(still fan-out prone)* | `2.8%` | `4.5%` | `$138,000 / mo` ✕ | ⛔ Finds tables better, but still hallucinates multi-grain joins |
| **🟦 Tier 3: Fine-Tuned SQL Code-LLM on Query Logs** | Trains custom weights on historical analyst SQL logs | `84.0%` | `2.1%` | `3.8%` | `$152,000 / mo` ✕ | ⛔ Memorizes deprecated schemas & legacy analyst mistakes |
| **🟩 Tier 4: Governed Semantic Layer (`Cube / dbt MetricFlow / Fabric`) + `SQLGlot AST & Dry-Run` Gate + OAuth `RLS`** | **LLM emits `{metrics, dims, filters}`; deterministic compiler builds symmetric-aggregate CTEs + `10GB` scan cap** | **`98.6%` 🏆** *(matches BI 100%)* | **`0.0%` 🏆** *(AST + Dry-Run cap)* | **`0.0%` 🏆** *(OAuth pass-through)* | **`$26,000 / mo` 🏆** *(-87.9% vs Tier 1)* | 🌟 **Recommended Enterprise Conversational BI Standard** |

> 💡 **Golden Rules for Enterprise Conversational BI Architecture**:
> 1. **Replace `Text-to-SQL` with `Text-to-Semantic-Intent` (`dbt MetricFlow / Cube / Fabric Semantic Model`)**: **"Never let an LLM invent join paths or financial formulas on the fly!"** For `90%` of business queries, the LLM outputs a declarative JSON query (`{ "measures": ["net_arr", "ticket_count"], "dimensions": ["region"] }`). The **Deterministic Semantic Layer Compiler** generates symmetric-aggregate CTEs that pre-aggregate each fact table at its native grain before joining—eliminating 1-to-N fan-out inflation and lifting KPI accuracy from `58.5%` to **`98.6%`**!
> 2. **`SQLGlot AST` Linter + `15ms EXPLAIN / DRY_RUN` Byte-Cap Circuit Breaker (`<= 10GB`)**: For the `10%` of ad-hoc exploratory queries that fall back to SQL, parse the query AST with `SQLGlot` (requiring read-only `SELECT`, mandatory partition filters, and no `CROSS JOIN`) and execute a zero-cost `EXPLAIN / DRY_RUN` check that blocks any query estimated to scan **`> 10GB`** (`0.0%` runaway scans!).
> 3. **End-to-End OAuth Identity Pass-Through (`On-Behalf-Of RLS`)**: Execute every semantic and warehouse query using the caller's Entra ID / OAuth identity so warehouse Row-Level Security (`RLS`) and Column Dynamic Masking apply automatically (`0.0%` unauthorized data exposure)!

---

## ② 🎯 Customer & Architect Deep-Dive Q&A (3 Join Fan-Out Inflation, BI Dashboard Parity & Warehouse Bill Explosions Pain Points)

### Q1: Why do frontier LLMs fail `41.5%` of the time when writing raw SQL across multiple enterprise fact tables—and how does a Semantic Layer Compiler achieve `98.6%`?
- **The Multi-Fact Chasm Trap (`1-to-N` Join Multiplication)**:
  - Joining `orders` (`1` row per order) with `shipments` (`3` rows per order) and `support_tickets` (`4` rows per customer) in a single flat `SELECT SUM(order_revenue)` multiplies each order's revenue by $3 \times 4 = 12\times$!
  - In Tier 4, the LLM never writes `JOIN` clauses. It requests `measures: ["order_revenue", "ticket_count"]` from **`Cube` or `dbt MetricFlow`**, which deterministically compiles **isolated per-grain CTE subqueries** (`cte_orders` grouped by customer, `cte_tickets` grouped by customer) before joining on the shared dimension key—guaranteeing `0%` fan-out inflation!

### Q2: How do we guarantee that our AI chat agent never gives the CEO a different revenue number than our official PowerBI / Tableau / Looker executive dashboard?
- **Single Source of Truth via Shared Semantic Layer APIs**:
  - Instead of duplicating business logic inside LLM system prompts, connect both your BI dashboards and your AI Agent MCP Server to the **same governed semantic layer (`dbt MetricFlow`, `Cube`, or `Microsoft Fabric Semantic Model DAX API`)**. When Finance updates the definition of `Net Retention Rate` in git, both PowerBI and the AI agent reflect the change instantaneously!

### Q3: How do we prevent a single poorly phrased natural-language question from triggering a `$250`, 50TB full-table scan in Snowflake or BigQuery?
- **3-Stage Cost & Syntax Guardrail (`SQLGlot AST` + `DRY_RUN Byte Cap` + Isolated X-Small Warehouse)**:
  1. **`SQLGlot AST` Validation (`0.8ms`)**: Rejects queries lacking partition predicates (`WHERE event_date >= ...`) on billion-row fact tables and enforces `LIMIT 1000`.
  2. **Warehouse `DRY_RUN / EXPLAIN` Gate (`15ms`)**: Checks optimizer-estimated bytes scanned before execution; rejects queries exceeding `10GB` and asks the user to narrow the date/region filter.
  3. **Dedicated Resource Monitor**: Routes AI queries to an isolated `X-Small` compute pool with a hard `15-second` query timeout (`0.0%` runaway bills).

---

## ③ 🧭 Visual Architecture Diagrams (Decision Tree + Semantic Layer & `SQLGlot AST` Topology + End-to-End Sequence)

### 3.1 Enterprise Conversational BI & Text-to-SQL Architecture Decision Tree

```mermaid
flowchart TD
    A["Enterprise Conversational BI Selection<br/>(800 Warehouse Tables | 200K Monthly Queries | Snowflake / Databricks / Fabric)"] --> B{"How are natural-language questions translated into warehouse queries?"}
    B -- "Raw physical DDL in prompt -> LLM writes free-form SQL (Tier 1)" --> C["⛔ 41.5% Wrong KPI Numbers + 8.2% Runaway Table Scans!<br/>1-to-N join fan-out inflates revenue & leaks 12.4% RLS rows ($215K/mo)"]
    B -- "Vector Schema-RAG + few-shot SQL examples only (Tier 2)" --> D["⛔ 19.4% Multi-Grain Join Errors!<br/>Drifts from official BI dashboard definitions ($138K/mo)"]
    B -- "Fine-tuned SQL Code-LLM on historical query logs (Tier 3)" --> E["⛔ Memorizes Deprecated Schemas & Analyst Bugs!<br/>84.0% accuracy & breaks on schema evolution ($152K/mo)"]
    B -- "Governed Semantic Layer (Cube/MetricFlow) + SQLGlot AST Gate + OAuth RLS (Tier 4)" --> F["🟩 Tier 4: Text-to-Semantic-Intent + Symmetric CTE Compiler + AST/RLS Gate 🏆<br/>(98.6% KPI Accuracy | 0.0% Runaway Scans | 0.0% RLS Leaks | -87.9% TCO)"]
    style F fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#f8fafc
    style C fill:#7f1d1d,stroke:#ef4444,color:#fecaca
    style D fill:#7f1d1d,stroke:#ef4444,color:#fecaca
    style E fill:#7f1d1d,stroke:#ef4444,color:#fecaca
```

### 3.2 Tier 4 Governed Semantic Layer (`Cube / MetricFlow`), `SQLGlot AST` Gate & OAuth `RLS` Reference Architecture

```mermaid
flowchart LR
    subgraph Auth["1. OAuth Identity & Intent Router"]
        User["👤 Executive / Analyst<br/>(Entra ID OAuth Token)"] --> Route{"🧭 Covered by Semantic Layer?"}
    end

    subgraph Sem["2A. Governed Semantic Path (90% KPI Queries)"]
        Route -- "90% Governed KPIs" --> JSON["🧠 LLM Emits Semantic JSON:<br/>{measures: [net_arr, tickets],<br/>dimensions: [region], time: Q3}"]
        JSON --> Comp["📐 Symmetric Aggregate Compiler<br/>(Cube / dbt MetricFlow / Fabric)<br/>Builds per-grain CTEs (0% fan-out!)"]
    end

    subgraph Adhoc["2B. Guarded Ad-Hoc SQL Path (10% Long-Tail)"]
        Route -- "10% Ad-Hoc" --> Gate["🛡️ SQLGlot AST + 15ms Dry-Run Gate<br/>• Requires partition date filter<br/>• Blocks scans > 10GB (0% runaway)"]
    end

    subgraph Exec["3. OAuth RLS & Data Warehouse"]
        Comp & Gate --> RLS["🔐 Caller OAuth RLS Pass-Through<br/>(0.0% unauthorized row exposure)"]
        RLS --> DW[("❄️ Snowflake / Databricks / Fabric<br/>(100% parity with BI dashboards)")]
    end
```

### 3.3 End-to-End Sequence: Multi-Fact Symmetric Aggregation, OAuth `RLS` Enforcement & `10GB` Scan Guardrail

```mermaid
sequenceDiagram
    autonumber
    participant VP as Regional Sales VP (OAuth: EMEA)
    participant Agent as Conversational BI Gateway
    participant Sem as Semantic Layer (Cube / MetricFlow)
    participant Gate as SQLGlot AST & Dry-Run Guard
    participant DW as Data Warehouse (Snowflake / Fabric)

    VP->>Agent: 1. Ask: `"Compare Q3 Net ARR and P1 Support Tickets by Country"`
    Agent->>Sem: 2. LLM emits semantic JSON: `measures=["net_arr", "p1_tickets"], dims=["country"], date="2026-Q3"`
    Sem->>Sem: 3. Compile into two isolated grain-aligned CTEs (`cte_arr` + `cte_tickets`) joined on `country` (`0%` fan-out inflation!)
    Sem->>Sem: 4. Bind caller's OAuth token -> inject `WHERE region = 'EMEA'` (`0.0%` cross-region RLS leak)
    Sem->>Gate: 5. Verify partition filter `2026-Q3` & `EXPLAIN` estimated scan = `180 MB (<= 10GB cap)`
    Gate->>DW: 6. Execute query & return exact table + chart + Metric Lineage Pill (`98.6%` accuracy, `340ms`)!
```

---

## ④ 📊 4-Tier Architecture Decision Matrix

| Evaluation Dimension | 🟥 Tier 1: Raw `DDL` Free-Form SQL | 🟨 Tier 2: `Schema-RAG` Only | 🟦 Tier 3: Fine-Tuned SQL LLM | 🟩 Tier 4: Semantic Layer (`Cube/MetricFlow`) + `AST` Gate + `RLS` |
|---|---|---|---|---|
| **Executive KPI Accuracy** | `58.5%` ✕ (join fan-out bugs) | `80.6%` ✕ (multi-grain drift) | `84.0%` | 🏆 **`98.6%` (deterministic symmetric-aggregate compiler)** |
| **Runaway Full-Table Scans** | `8.2%` ✕ (50TB scan spikes) | `2.8%` | `2.1%` | 🏆 **`0.0%` (`SQLGlot AST` partition gate + `10GB` dry-run cap)** |
| **`RLS` Unauthorized Exposure** | `12.4%` ✕ (shared DB account) | `4.5%` | `3.8%` | 🏆 **`0.0%` (end-to-end OAuth pass-through & semantic RLS)** |
| **BI Dashboard Metric Parity** | None (divergent SQL) | Manual prompt sync | Stale weights | 🏆 **`100%` identical (shares semantic model with PowerBI/Tableau)** |
| **200K-Query Monthly TCO** | `$215,000 / mo` ✕ | `$138,000 / mo` ✕ | `$152,000 / mo` ✕ | **`$26,000 / mo` 🏆 (-87.9% vs Tier 1)** |

---

## ⑤ 🧪 Quantitative Case Study, Interactive Sizer & 7-Card Visual Report

- **Interactive HTML Sizer**: 👉 [Open Enterprise Conversational BI: Text-to-SQL vs. Semantic Layer Sizer](../reports/enterprise-conversational-bi-text-to-sql-semantic-layer-metricflow-sizer.html)
- **7-Card 4:5 Visual Benchmark Report**: 👉 [Open 7-Card Visual Benchmark Deck](../reports/enterprise-conversational-bi-text-to-sql-semantic-layer-metricflow-selection-cards.html)
- **Deterministic Verification**:
  ```bash
  python experiments/enterprise_conversational_bi_text_to_sql_semantic_layer_metricflow_selection/run.py --check
  python -m unittest tests/test_enterprise_conversational_bi_text_to_sql_semantic_layer_metricflow_selection.py
  ```

---

## ⑥ 🛠️ Production Readiness & Architecture Review Checklist

- [ ] **Route KPI Questions Through a Governed Semantic Layer (`Text-to-Semantic-Intent` on `Cube / dbt MetricFlow / Fabric Semantic Model`)**: Never allow an LLM to improvise multi-fact `JOIN` paths or financial aggregation formulas from raw physical DDLs.
- [ ] **Enforce End-to-End OAuth Identity Pass-Through (`On-Behalf-Of RLS`)**: Prohibit shared superuser database connections; evaluate row-level security and column masking under the querying user's identity.
- [ ] **Validate Ad-Hoc SQL with a `SQLGlot AST` Linter**: Block all DDL/DML mutations, `CROSS JOIN`s, and queries against partitioned fact tables that omit a partition filter.
- [ ] **Gate Execution Behind a `15ms EXPLAIN / DRY_RUN` Byte Limit (`<= 10GB`) and a Dedicated `X-Small` Compute Pool**: Eliminate `$250` runaway full-table scans before warehouse compute is billed.
- [ ] **Attach a Deterministic "Metric Lineage Pill" to Every Response**: Display the governed metric name, time grain, and applied filters alongside the chart so executives can verify parity with official BI dashboards.
