# Enterprise Complex Excel & Multi-Header Financial Table RAG (`openpyxl AST` Merged-Cell Un-Merging, Hierarchical Header Propagation & Embedded `DuckDB` SQL Execution) Selection Guide

> **Language / 语言**: **English (Current)** | [中文](./enterprise-excel-complex-financial-table-rag-header-propagation-duckdb-selection-faq.md)  
> **Scope**: End-to-end production reference architecture and sizing playbook for Cloud & AI Solution Architects, Financial Data Engineers (`FP&A / Procurement / Actuarial`), and RAG Platform Leads indexing complex enterprise `.xlsx` workbooks (`100,000+` workbooks, `1.2M` sheets).  
> **Provenance**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`). Deterministic evaluation model sealed in [`experiments/enterprise_excel_complex_financial_table_rag_header_propagation_duckdb_selection/results.json`](../../experiments/enterprise_excel_complex_financial_table_rag_header_propagation_duckdb_selection/results.json).

![Enterprise Complex Excel & Multi-Header Financial Table RAG Summary](../../assets/svg/enterprise-excel-complex-financial-table-rag-header-propagation-duckdb-selection-faq-summary.svg)

---

## ① 📌 Executive Summary (3-Minute Architect Verdict)

Across finance (`FP&A`), supply chain, and procurement, `60%+` of authoritative business data lives inside **complex multi-sheet `.xlsx` workbooks (`100,000` workbooks, `1.2M` sheets, `600,000` monthly queries)** featuring **3-level merged column headers (`FY2026 -> Q3 -> Actuals ($ in Thousands)`), indented parent-child row hierarchies (`OPEX -> R&D -> Cloud Compute`), side-by-side sub-tables on a single sheet, and `=SUM()` subtotal rows**. Three common RAG architectures fail on real-world spreadsheets:
1. **Trap #1: Naive `pandas.read_excel().to_csv()` + 500-Token Text Chunking (`Tier 1`, Only `56.5%` QA Accuracy — `43.5%` Failure Rate, `38.4%` Arithmetic & Unit-Scale Error Rate, `$195,000/month` TCO)**:
   - Merged cells in rows `1..4` turn into `Unnamed: 1, Unnamed: 2, NaN`!
   - When a 300-row table is sliced into 500-token chunks, **Chunk #4 (Rows `150–200`) contains only bare numbers with zero column headers and loses the top-of-sheet banner `"($ in Thousands)"`**—causing `1,000x` unit errors and mental-math hallucinations!
2. **Trap #2: Rendering Excel Sheets as PNG Screenshots for a Vision LLM (`VLM Screenshot RAG`, `Tier 2`, `72.0%` Accuracy, Fails on 500-Row x 40-Col Sheets, Costs `$210,000/month` in Vision Tokens)**.
3. **Trap #3: Dumping Raw `.xlsx` Files into a Python `Pandas REPL` Without Coordinate-AST Normalization (`Tier 3`, `78.5%` Accuracy, `4,800ms` P95 Latency as the Agent Loops on `KeyError: Unnamed: 3`, `$138,000/month` TCO)**.

Evaluated across **100,000 enterprise Excel workbooks (`1.2M` worksheets, `600,000` monthly spreadsheet RAG queries, `68%` merged multi-headers)**, the four architectures compare as follows:

| Table RAG Tier | Header Un-Merging, Row Context & Math Stack | Complex Table QA Accuracy | Math & Unit-Scale Error | Query `P95` Latency | 600K-Query Monthly TCO | Production Verdict |
|---|---|---|---|---|---|---|---|
| **🟥 Tier 1: Naive `read_excel().to_csv()` + 500t Chunks** | Loses merged headers (`Unnamed: 1`) & orphans rows #50+ | `56.5%` ✕ *(43.5% wrong answers)* | `38.4%` ✕ *(1,000x unit errors)* | `820.0 ms` | `$195,000 / mo` ✕ *(severe financial errors)* | ⛔ Never chunk spreadsheets as raw flat CSV text |
| **🟨 Tier 2: Vision `VLM` Sheet Screenshot RAG** | Renders sheets to images for multimodal LLM | `72.0%` ✕ *(unreadable on wide sheets)* | `24.5%` ✕ *(visual sum hallucinations)* | `3,400.0 ms` ✕ | `$210,000 / mo` ✕ *(massive VLM token cost)* | ⛔ Vision models cannot read 500-row tables or sum 80 rows reliably |
| **🟦 Tier 3: Raw `.xlsx` into `Pandas REPL` Loop** | LLM writes iterative Python scripts on uncleaned sheets | `78.5%` | `14.2%` | **`4,800.0 ms` ✕** *(multi-turn debug lag)* | `$138,000 / mo` ✕ | ⛔ Side-by-side sub-tables and merged cells break `df.columns` |
| **🟩 Tier 4: `openpyxl` Coordinate-AST Un-Merging + Header/Unit Propagation + Embedded `DuckDB` SQL** | **Splits sub-table islands, flattens merged headers (`FY26_Q3_Actual_USD_K`), propagates row trees & runs `8ms` `DuckDB` SQL** | **`97.8%` 🏆** *(+41.3 pts vs Tier 1)* | **`0.2%` 🏆** *(deterministic SQL + unit scale)* | **`390.0 ms` 🏆** *(12.3x faster than T3)* | **`$23,000 / mo` 🏆** *(-88.2% vs Tier 1)* | 🌟 **Recommended Enterprise Complex Excel & Financial Table RAG Standard** |

> 💡 **Golden Rules for Enterprise Complex Excel & Financial Table RAG**:
> 1. **`openpyxl` Coordinate-AST Sub-Table Splitting & Merged-Cell Un-Merging**: **"Never pass a raw `.xlsx` directly to `pandas.read_excel()`!"** At ingestion time, parse the 2D worksheet grid with `openpyxl`:
>    - **Split Sub-Table Islands**: Detect blank-row/blank-column and border bounding boxes to split side-by-side or stacked sub-tables on a single worksheet into independent tables.
>    - **Broadcast Merged Ranges & Flatten Header Stacks**: Fill every cell in `ws.merged_cells.ranges` with its top-left value and join multi-row headers vertically into canonical column names (`FY2026__Q3_Actuals__USD_Thousands`)—eliminating `Unnamed: 1` forever!
> 2. **Propagate Indented Row Hierarchies, Unit Multipliers (`$K / $M`), and `is_subtotal` Flags**: Reconstruct parent-child row indentation (`OPEX > R&D > Cloud`) into a dedicated `row_hierarchy_path` column, attach the sheet's unit scale (`scale=1000`), and inspect `data_only=False` formulas to tag `=SUM() / =SUBTOTAL()` rows with **`is_subtotal = true`** (preventing `2x` double-counting during SQL `SUM()` queries!).
> 3. **Dual-Engine Execution (`Self-Contained Row Cards` for Lookup + Embedded `DuckDB / Parquet` for Math)**: Store self-contained row-tuple cards (with exact `[Workbook#Sheet!R142C5]` coordinates) in the hybrid vector index for entity discovery, and execute all filtering, YoY ratios, and aggregations via **`8ms` Embedded `DuckDB` SQL (`0.2%` math error)**!

---

## ② 🎯 Customer & Architect Deep-Dive Q&A (3 Fixing `Unnamed: 1`, Preventing `2x` Subtotal Double-Counting & Preserving Cross-Sheet Formulas Pain Points)

### Q1: How does Tier 4 prevent the `Unnamed: 1` column corruption and Row-#150 header orphaning that cause `43.5%` of Tier-1 Excel RAG failures?
- **Coordinate-AST Header Tree Flattening & Self-Contained Row Tuples**:
  - When `openpyxl` un-merges `B2:E2 ("FY2026")` and `B3:C3 ("Q3 Actuals")`, the AST compiler concatenates the vertical header stack for Column B into `"FY2026__Q3_Actuals"` and extracts the sheet banner `"($ in Thousands)"` as `unit_multiplier = 1000`.
  - Every indexed chunk or row-tuple serializes `{row_path: "EMEA > Germany > Enterprise", FY2026__Q3_Actuals_USD_K: 1420.5, cell_ref: "P&L!B154"}`—so no row is ever separated from its column header, parent category, or currency scale!

### Q2: Why do standard SQL / Pandas agents frequently double-count financial totals by `2x` or `3x` on Excel P&L sheets?
- **Tagging `=SUM()` and `=SUBTOTAL()` Rows via Dual-Pass `openpyxl` Inspection**:
  - Financial spreadsheets interleave detail rows (`Cloud Compute`, `SaaS Licenses`) with bold summary rows (`Total R&D = SUM(D10:D18)`). If an agent runs `SELECT SUM(q3_actual) FROM sheet`, it sums both the detail rows AND the subtotal rows!
  - By reading the workbook once with `data_only=False` (to inspect raw formula strings `=SUM(...)`) and once with `data_only=True` (to read cached values), Tier 4 stamps **`is_subtotal = true`** on every formula-aggregated row—allowing `DuckDB` queries (`WHERE NOT is_subtotal`) to compute `100%` exact totals!

### Q3: Why did Tier 3 (`Raw .xlsx into Pandas REPL`) suffer a `4,800ms` P95 latency and `21.5%` failure rate?
- **Eliminating Runtime Trial-and-Error Schema Discovery**:
  - Asking an LLM at query time to write Python code that guesses which row contains the header (`header=3`? `header=[4,5]`?) and where sub-tables start wastes 3–4 round-trip REPL exceptions (`4.8s`). Doing Coordinate-AST normalization **once at ingestion time** lets the query-time LLM write a clean 1-line `DuckDB` query over pre-flattened column names in **`390ms`**!

---

## ③ 🧭 Visual Architecture Diagrams (Decision Tree + `openpyxl AST + DuckDB` Dual-Index Topology + Execution Sequence)

### 3.1 Enterprise Complex Excel & Financial Table RAG Decision Tree

```mermaid
flowchart TD
    A["Enterprise Complex Excel & Financial Table RAG Selection<br/>(100K Workbooks | 1.2M Sheets | 68% Merged Multi-Headers | 600K Queries/Mo)"] --> B{"How are complex multi-header spreadsheets parsed, indexed, and queried?"}
    B -- "Naive read_excel().to_csv() + 500-token text chunks (Tier 1)" --> C["⛔ 56.5% QA Accuracy + 38.4% Math/Unit Errors!<br/>Merged cells become Unnamed:1 & rows #150+ lose headers ($195K/mo)"]
    B -- "Render sheets as PNG images for Vision VLM (Tier 2)" --> D["⛔ 72.0% Accuracy & $210K/mo VLM Token Bill!<br/>Fails on 500-row wide sheets & hallucinates sums (24.5% error)"]
    B -- "Raw uncleaned .xlsx into Python Pandas REPL loop (Tier 3)" --> E["⛔ 4,800ms P95 Latency from KeyError Retry Loops!<br/>Multi-table sheets & merged cells break df.columns ($138K/mo)"]
    B -- "openpyxl Coordinate-AST Un-Merging + Header Propagation + DuckDB SQL (Tier 4)" --> F["🟩 Tier 4: openpyxl Coordinate-AST + Self-Contained Cards + DuckDB 🏆<br/>(97.8% Accuracy | 0.2% Math Error | 390ms P95 | $23K/mo, -88.2% TCO)"]
    style F fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#f8fafc
    style C fill:#7f1d1d,stroke:#ef4444,color:#fecaca
    style D fill:#7f1d1d,stroke:#ef4444,color:#fecaca
    style E fill:#7f1d1d,stroke:#ef4444,color:#fecaca
```

### 3.2 Tier 4 `openpyxl` Coordinate-AST Un-Merging, Subtotal Tagging & Dual `Card + DuckDB` Architecture

```mermaid
flowchart LR
    subgraph AST["1. Ingestion openpyxl Coordinate-AST Normalizer"]
        XLS["📊 Complex .xlsx Workbook<br/>(Merged headers, sub-tables, formulas)"] --> Split["🏝️ Sub-Table Island Splitter<br/>• Isolates multiple tables/sheet<br/>• Tags =SUM() rows (is_subtotal=true)"]
        Split --> Flat["🧱 Merged-Cell Broadcast & Header Stack<br/>B2:E2 + B3:C3 -> Flattened Column:<br/>'FY26_Q3_Actual_USD_Thousands'"]
        Flat --> Tree["🌲 Indented Row-Hierarchy Propagator<br/>Injects 'OPEX > R&D > Cloud' + [Sheet!R42C5]"]
    end

    subgraph Store["2. Dual-Modal Storage Engine"]
        Tree --> Card[("🗂️ Self-Contained Row-Tuple Cards<br/>(Hybrid BM25 + Vector)<br/>Every row carries full header & unit scale")]
        Tree --> Duck[("🦆 Embedded DuckDB / Parquet Tables<br/>Excludes is_subtotal=true;<br/>executes exact SUM / YoY SQL in 8ms!")]
    end

    subgraph Route["3. Query Execution (390ms P95, 97.8% Accuracy)"]
        Q["👤 Analyst Query"] --> Exec["🧭 Schema Match + 8ms DuckDB SQL"]
        Exec --> Card & Duck
    end
```

### 3.3 End-to-End Sequence: Answering a Multi-Header P&L Variance Query with Subtotal Filtering & Unit Scaling in `390ms`

```mermaid
sequenceDiagram
    autonumber
    participant CFO as FP&A Analyst / CFO
    participant Router as Table RAG & DuckDB Orchestrator
    participant VDB as Hybrid Table Schema & Row-Card Index
    participant Duck as Embedded DuckDB Engine (Parquet)

    CFO->>Router: 1. Query: `"In the FY26 Budget Workbook, how much did Q3 EMEA R&D sub-items exceed budget in USD Millions?"`
    Router->>VDB: 2. Retrieve normalized table schema `Sheet3_Table1`: columns `[row_path, FY26_Q3_EMEA_Actual_USD_K, FY26_Q3_EMEA_Budget_USD_K, is_subtotal]`
    Router->>Duck: 3. Execute SQL: `SELECT SUM(FY26_Q3_EMEA_Actual_USD_K - FY26_Q3_EMEA_Budget_USD_K) / 1000.0 FROM tbl WHERE row_path LIKE 'OPEX > R&D > %' AND NOT is_subtotal`
    Note over Router,Duck: Automatically excludes `is_subtotal=true` summary rows (0% double-counting!) and converts `$K -> $M`!
    Duck-->>Router: 4. `8ms`: Return exact variance `+$1.4285M` + source cell range `[Sheet3!D48:E56]`
    Router-->>CFO: 5. Stream verified answer with exact cell citations & formula lineage (`390ms` total, `0.2%` error rate)!
```

---

## ④ 📊 4-Tier Architecture Decision Matrix

| Evaluation Dimension | 🟥 Tier 1: `to_csv()` + Chunks | 🟨 Tier 2: Vision `VLM` Images | 🟦 Tier 3: Raw `.xlsx` Pandas | 🟩 Tier 4: `openpyxl AST` + `DuckDB` |
|---|---|---|---|---|
| **Complex Table QA Accuracy** | `56.5%` ✕ (`Unnamed: 1` & orphans) | `72.0%` ✕ (fails on >50 rows) | `78.5%` | 🏆 **`97.8%` (un-merged header stack + row tree propagation)** |
| **Math & Unit-Scale Error Rate** | `38.4%` ✕ (`1,000x` scale errors) | `24.5%` ✕ (visual math errors) | `14.2%` | 🏆 **`0.2%` (`8ms` `DuckDB` SQL + `is_subtotal` filter)** |
| **Query `P95` Latency** | `820.0 ms` | `3,400.0 ms` ✕ | `4,800.0 ms` ✕ (REPL debug loops) | 🏆 **`390.0 ms` (12.3x faster than iterative Pandas REPL)** |
| **Exact Cell-Coordinate Citation** | No (lost in text chunks) | Sheet image only | Brittle | 🏆 **Exact `[Workbook#Sheet!R48C5]` cell provenance** |
| **600K-Query Monthly TCO** | `$195,000 / mo` ✕ | `$210,000 / mo` ✕ | `$138,000 / mo` ✕ | **`$23,000 / mo` 🏆 (-88.2% vs Tier 1)** |

---

## ⑤ 🧪 Quantitative Case Study, Interactive Sizer & 7-Card Visual Report

- **Interactive HTML Sizer**: 👉 [Open Enterprise Complex Excel & Financial Table RAG Sizer](../reports/enterprise-excel-complex-financial-table-rag-header-propagation-duckdb-sizer.html)
- **7-Card 4:5 Visual Benchmark Report**: 👉 [Open 7-Card Visual Benchmark Deck](../reports/enterprise-excel-complex-financial-table-rag-header-propagation-duckdb-selection-cards.html)
- **Deterministic Verification**:
  ```bash
  python experiments/enterprise_excel_complex_financial_table_rag_header_propagation_duckdb_selection/run.py --check
  python -m unittest tests/test_enterprise_excel_complex_financial_table_rag_header_propagation_duckdb_selection.py
  ```

---

## ⑥ 🛠️ Production Readiness & Architecture Review Checklist

- [ ] **Ban Naive `pandas.read_excel().to_csv()` on Multi-Header Business Workbooks**: Use `openpyxl` to un-merge `ws.merged_cells.ranges` and concatenate multi-row header stacks into descriptive column names (`FY26__Q3__Actual_USD_K`).
- [ ] **Split Multi-Table Worksheets via Bounding-Box Island Detection**: Isolate side-by-side or vertically stacked sub-tables on a single sheet before schema extraction.
- [ ] **Propagate Indented Row Hierarchies (`OPEX > R&D > Cloud`) and Sheet Unit Multipliers (`$ in Thousands`) into Every Row**: Guarantee that row #250 never loses its parent category or unit scale.
- [ ] **Inspect `data_only=False` Formulas to Tag `=SUM() / =SUBTOTAL()` Summary Rows (`is_subtotal = true`)**: Prevent `2x` double-counting when executing SQL `SUM()` aggregations.
- [ ] **Register Normalized Tables in Embedded `DuckDB / Parquet` Alongside Self-Contained Semantic Row Cards**: Route all arithmetic, filtering, and YoY comparisons to `8ms` `DuckDB` SQL (`0.2%` error rate).
