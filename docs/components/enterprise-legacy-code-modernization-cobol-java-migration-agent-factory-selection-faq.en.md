# Enterprise Legacy Application Modernization Factory (`COBOL / PL/SQL / Java 8 -> Cloud-Native Java 21`), `AST` Call-Graph Topological Slicing, Golden I/O Equivalence Verification & Strangler-Fig Shadowing Architecture Selection Guide

> **Language / 语言**: **English (Current)** | [中文](./enterprise-legacy-code-modernization-cobol-java-migration-agent-factory-selection-faq.md)  
> **Scope**: End-to-end production reference architecture and sizing playbook for Cloud & AI Solution Architects, Core Banking / Insurance / ERP Modernization Leads, and Principal Engineers migrating large-scale legacy codebases (`500,000+ LOC` of `COBOL / CICS / DB2 PL/SQL / Java 8`) to cloud-native microservices.  
> **Provenance**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`). Deterministic evaluation model sealed in [`experiments/enterprise_legacy_code_modernization_cobol_java_migration_agent_factory_selection/results.json`](../../experiments/enterprise_legacy_code_modernization_cobol_java_migration_agent_factory_selection/results.json).

![Enterprise Legacy Code Modernization Factory Summary](../../assets/svg/enterprise-legacy-code-modernization-cobol-java-migration-agent-factory-selection-faq-summary.svg)

---

## ① 📌 Executive Summary (3-Minute Architect Verdict)

When enterprises attempt to modernize **`500,000+` lines of legacy `COBOL / CICS`, `Oracle/DB2 PL/SQL`, or monolithic `Java 8` code (`420` interconnected modules)** into cloud-native `Java 21 Spring Boot + PostgreSQL`, three naive AI approaches lead to costly project failures:
1. **Trap #1: Naive Whole-File LLM Prompt Translation (`Tier 1`, `44.0%` Compile/Dependency Failure Rate, `31.5%` Escaped Fixed-Point Decimal & Cursor Bugs — Only `68.5%` I/O Equivalence, `$620,000` TCO)**:
   - Legacy mainframe `COBOL` uses `PIC S9(7)V99 COMP-3` packed decimals that **truncate toward zero (`RoundingMode.DOWN`)** by default unless `ROUNDED` is specified, and sorts strings in **`EBCDIC` order (`'a' < 'A' < '0'`)** rather than `ASCII ('0' < 'A' < 'a')`. A raw LLM translates `COMP-3` into Java `double` or `HALF_UP`—causing **1-cent financial discrepancies across millions of accounts**!
2. **Trap #2: Manual IDE Copilot Snippet-by-Snippet Rewriting (`Tier 2`, Takes `14.5 Months` per `500K LOC`, `12.8%` Escaped Bug Rate, `$490,000` TCO)**: Developers spend 75% of their time manually reverse-engineering undocumented 30-year-old business rules.
3. **Trap #3: Letting the Same LLM Write Both the New Code AND the JUnit Tests ("Self-Graded Homework", `Tier 3`, `14.2%` Escaped Logic Bugs, `$340,000` TCO)**:
   - When an LLM misunderstands a legacy `COBOL PERFORM` or `PL/SQL` cursor edge case, **it bakes that exact same misunderstanding into the unit test assertions (`assertEquals`)**—producing a `100%` green test suite that fails immediately in production!

Evaluated across a **500,000-LOC legacy migration (`420` modules/copybooks, `10,000` production golden I/O test vectors)**, the four modernization architectures compare as follows:

| Modernization Tier | Slicing, Type Mapping & Verification Stack | Functional I/O Equivalence | Escaped Logic / Decimal Bugs | 500K-LOC Timeline | Total Migration TCO | Production Verdict |
|---|---|---|---|---|---|---|---|
| **🟥 Tier 1: Naive Whole-File LLM Prompt Translation** | Dumps raw legacy files into LLM; no AST DAG or golden replay | `68.5%` ✕ *(compile & math bugs)* | `31.5%` ✕ *(COMP-3 penny drift)* | `11.0 Months` *(manual bug-fix bog)* | `$620,000` ✕ *(massive rework)* | ⛔ Never use unconstrained whole-file prompt translation on core logic |
| **🟨 Tier 2: Manual IDE Copilot Snippet Rewrite** | Human engineers manually rewrite & test file by file | `87.2%` | `12.8%` ✕ | `14.5 Months` ✕ *(too slow)* | `$490,000` ✕ *(high labor cost)* | ⛔ Cannot scale across multi-million-line legacy estates |
| **🟦 Tier 3: Agentic Code + LLM Self-Generated Unit Tests** | Same LLM generates Java code and its own JUnit assertions | `85.8%` | **`14.2%` ✕** *(self-graded tests!)* | `5.5 Months` | `$340,000` ✕ | ⛔ Correlated hallucination: LLM writes its own bugs into the test assertions |
| **🟩 Tier 4: `AST` Leaf-First Topological Slicer + `COMP-3 BigDecimal` Guard + `10K Golden I/O Tape` Harness + Shadowing** | **`ProLeap/Tree-sitter` leaf-first DAG + recorded mainframe I/O replay loop + Strangler-Fig shadowing** | **`99.8%` 🏆** *(byte-level parity)* | **`0.2%` 🏆** *(-31.3 pts vs Tier 1)* | **`2.4 Months` 🏆** *(6.0x faster!)* | **`$85,000` 🏆** *(-86.3% vs Tier 1)* | 🌟 **Recommended Enterprise Legacy Modernization Factory Standard** |

> 💡 **Golden Rules for an Enterprise AI Code Modernization Factory**:
> 1. **`AST` Call-Graph Leaf-First Topological Slicing (`ProLeap / ANTLR / Tree-sitter`)**: **"Never translate callers before their leaf dependencies!"** Build the complete cross-module Call & Data Dependency DAG first. Deterministically compile `COPYBOOK` structs into typed Java `Record`s, then migrate leaf utility paragraphs upward so every parent module compiles against verified interfaces (`99.4%` first-pass compile rate).
> 2. **Enforce Deterministic `COMP-3 -> BigDecimal` & `EBCDIC` Runtime Shims**: Ban `float` and `double` in the target AST linter; map packed decimals to exact scale and truncation semantics (`RoundingMode.DOWN` vs. `HALF_UP`).
> 3. **Never Let the LLM Grade Its Own Homework — Use a `10,000-Vector` Recorded Production `Golden I/O Tape` Harness**: Replay 10,000 real historical input/output pairs captured from the legacy mainframe (`VSAM / CICS / DB2`) against the generated Java 21 code. Feed any `$0.01` field discrepancy automatically back into the **Repair Agent** until `10,000 / 10,000` golden vectors match (`99.8%` equivalence)!

---

## ② 🎯 Customer & Architect Deep-Dive Q&A (3 `COMP-3` Penny Discrepancies, Why Self-Generated Unit Tests Lie & Slicing 18,000-Line Monoliths Pain Points)

### Q1: Why does translating `COBOL` or `PL/SQL` financial code with a frontier LLM cause `31.5%` arithmetic and sorting discrepancies even when the control flow looks identical?
- **Packed-Decimal Truncation (`COMP-3`), High-Order Overflow, and `EBCDIC` Collation**:
  - In COBOL, `COMPUTE INTEREST = PRINCIPAL * RATE / 365` without the `ROUNDED` keyword **truncates at the declared decimal scale (`RoundingMode.DOWN`)**, whereas Java developers and LLMs default to `double` or `HALF_UP`.
  - Similarly, `EBCDIC` sorts lowercase before uppercase before digits (`'a' < 'A' < '0'`), opposite to `ASCII/UTF-8`. Tier 4 enforces a deterministic `CobolRuntimeShim` library and AST linter that preserves exact mainframe arithmetic and collation semantics!

### Q2: Why did `14.2%` of semantic bugs escape into production in Tier 3 even though the LLM-generated JUnit test suite had `95%` line coverage and `100%` passing tests?
- **Correlated Hallucination ("The LLM Grading Its Own Homework")**:
  - If the LLM misinterprets a legacy leap-year interest rule when writing `InterestService.java`, it uses that **exact same flawed mental model** when generating `InterestServiceTest.java`!
  - Only an **external, model-independent oracle**—a `10,000-vector` **Golden I/O Tape** recorded from actual legacy production runs plus property-based differential fuzzing—can catch semantic misinterpretations (`0.2%` escaped defect rate).

### Q3: How do we slice an 18,000-line spaghetti COBOL or PL/SQL file with 60 `PERFORM` paragraphs mutating shared global `WORKING-STORAGE` variables?
- **Data-Flow Read/Write Set Slicing (`Program Slicing AST`)**:
  - Instead of chunking by line count, Tier 4 computes the exact `READ_SET` and `WRITE_SET` of global variables for each `PERFORM` paragraph, automatically refactoring shared state into explicit typed input/output structs so each paragraph can be transpiled and verified in isolation!

---

## ③ 🧭 Visual Architecture Diagrams (Decision Tree + `AST + Golden I/O Tape Factory` Topology + Automated Differential Repair Sequence)

### 3.1 Enterprise Legacy Code Modernization Factory Decision Tree

```mermaid
flowchart TD
    A["Enterprise Legacy Code Modernization Selection<br/>(500K LOC COBOL / PL-SQL / Java 8 | 420 Modules | Core Banking / ERP / Telecom)"] --> B{"How are legacy modules sliced, transpiled, and verified for 100% parity?"}
    B -- "Raw whole-file LLM prompt translation (Tier 1)" --> C["⛔ 31.5% Escaped Decimal/Logic Bugs + 44% Compile Failures!<br/>COMP-3 rounding & EBCDIC bugs cause massive rework ($620K TCO)"]
    B -- "Manual IDE Copilot snippet-by-snippet rewrite (Tier 2)" --> D["⛔ 14.5-Month Timeline ($490K TCO)!<br/>Engineers spend 75% of time reverse-engineering undocumented rules"]
    B -- "Agentic code + LLM self-generated unit tests (Tier 3)" --> E["⛔ 14.2% Escaped Semantic Bugs!<br/>LLM bakes its own misinterpretations into JUnit assertions ($340K)"]
    B -- "AST Leaf-First Slicer + COMP-3 Guard + 10K Golden I/O Tape + Shadowing (Tier 4)" --> F["🟩 Tier 4: AST Topological Slicer + 10K Golden I/O Replay Factory 🏆<br/>(99.8% Equivalence | 2.4-Month Delivery | 0.2% Escaped Bugs | -86.3% TCO)"]
    style F fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#f8fafc
    style C fill:#7f1d1d,stroke:#ef4444,color:#fecaca
    style D fill:#7f1d1d,stroke:#ef4444,color:#fecaca
    style E fill:#7f1d1d,stroke:#ef4444,color:#fecaca
```

### 3.2 Tier 4 `AST` Leaf-First Slicer, Deterministic Decimal Shim & `10,000-Vector` Golden I/O Replay Factory Architecture

```mermaid
flowchart LR
    subgraph Slice["1. AST Call-Graph & Data-Flow Slicer"]
        Code["🏛️ 500K LOC Legacy Estate<br/>(COBOL / PL-SQL / Java 8)"] --> DAG["🌳 ProLeap / Tree-Sitter AST<br/>Builds Call & Read/Write Set DAG;<br/>sorts Leaf-First"]
    end

    subgraph Trans["2. Spec-First & Guarded Transpilation"]
        DAG --> Spec["📘 Business Rule Extractor<br/>Generates human-reviewed spec"]
        DAG & Spec --> Gen["⚙️ Transpiler Agent + AST Linter<br/>• Bans float/double<br/>• Maps COMP-3 -> BigDecimal(DOWN)<br/>• Preserves EBCDIC collation"]
    end

    subgraph Verify["3. Independent 10K Golden I/O Tape Harness"]
        Gen --> Replay["🧪 10,000 Production Golden Tapes<br/>Compares Legacy Output vs Java 21"]
        Replay -- "$0.01 Field Diff" --> Fix["🔧 Differential Repair Agent<br/>Iterates until 0 discrepancy"]
        Replay -- "10,000 / 10,000 Match" --> Shadow["🚀 Strangler-Fig Traffic Shadowing<br/>(99.8% Production Equivalence!)"]
    end
```

### 3.3 End-to-End Sequence: Catching and Auto-Repairing a `$0.01` `COMP-3` Truncation Discrepancy via Golden I/O Tapes

```mermaid
sequenceDiagram
    autonumber
    participant Slicer as AST Topological Scheduler
    participant Trans as Transpiler Agent (Java 21)
    participant Tape as 10K Golden I/O Replay Harness
    participant Repair as Differential Repair Agent
    participant Proxy as Strangler-Fig Shadow Proxy

    Slicer->>Trans: 1. Dispatch leaf module `INT-CALC-04` (`COMPUTE DAILY-INT = BAL * RATE / 365`) + typed `COPYBOOK` records
    Trans->>Tape: 2. Emit candidate Java 21 class (accidentally using `RoundingMode.HALF_UP`)
    Tape->>Tape: 3. Replay 10,000 mainframe golden records -> Vector `#412` (`bal=$10,000.55`): Mainframe=`$1.36` vs Java=`$1.37` (`+$0.01` diff!)
    Tape->>Repair: 4. Reject build! Send Vector `#412` input, expected `$1.36`, actual `$1.37` & COBOL line 284 to Repair Agent
    Repair->>Tape: 5. Detect missing `ROUNDED` keyword on COBOL line 284 -> patch Java division to `RoundingMode.DOWN`
    Tape->>Proxy: 6. Re-run all 10,000 golden vectors -> `10,000 / 10,000` exact match! Promote module to 14-day shadow traffic!
```

---

## ④ 📊 4-Tier Architecture Decision Matrix

| Evaluation Dimension | 🟥 Tier 1: Whole-File LLM | 🟨 Tier 2: Manual IDE Copilot | 🟦 Tier 3: LLM Self-Graded Tests | 🟩 Tier 4: `AST` Slicer + `10K Golden Tape` Factory |
|---|---|---|---|---|
| **Functional I/O Equivalence** | `68.5%` ✕ (decimal/cursor drift) | `87.2%` | `85.8%` | 🏆 **`99.8%` (verified against 10,000 production golden tapes)** |
| **Escaped Logic / Decimal Bugs** | `31.5%` ✕ (severe accounting bugs) | `12.8%` ✕ | `14.2%` ✕ (correlated bug in test) | 🏆 **`0.2%` (deterministic `COMP-3` shim + differential replay)** |
| **500K-LOC Migration Timeline** | `11.0 Months` (rework bog) | `14.5 Months` ✕ (slow) | `5.5 Months` | 🏆 **`2.4 Months` (6.0x faster than manual IDE rewrite)** |
| **Total 500K-LOC Migration TCO** | `$620,000` ✕ | `$490,000` ✕ | `$340,000` ✕ | **`$85,000` 🏆 (-86.3% vs Tier 1)** |

---

## ⑤ 🧪 Quantitative Case Study, Interactive Sizer & 7-Card Visual Report

- **Interactive HTML Sizer**: 👉 [Open Enterprise Legacy Code Modernization Factory Sizer](../reports/enterprise-legacy-code-modernization-cobol-java-migration-agent-factory-sizer.html)
- **7-Card 4:5 Visual Benchmark Report**: 👉 [Open 7-Card Visual Benchmark Deck](../reports/enterprise-legacy-code-modernization-cobol-java-migration-agent-factory-selection-cards.html)
- **Deterministic Verification**:
  ```bash
  python experiments/enterprise_legacy_code_modernization_cobol_java_migration_agent_factory_selection/run.py --check
  python -m unittest tests/test_enterprise_legacy_code_modernization_cobol_java_migration_agent_factory_selection.py
  ```

---

## ⑥ 🛠️ Production Readiness & Architecture Review Checklist

- [ ] **Build the Full `AST` Call & Data-Flow DAG (`ProLeap / ANTLR / Tree-sitter`) Before Translating Any Code**: Migrate in strict **Leaf-First Topological Order** starting from `COPYBOOK` data structures and zero-dependency utility modules.
- [ ] **Enforce Deterministic Mainframe Arithmetic (`COMP-3 -> BigDecimal`) and `EBCDIC` Collation Rules via AST Linters**: Prohibit native `float / double` types in financial or billing code generation.
- [ ] **Never Rely on LLM-Generated Unit Test Assertions Alone ("No Self-Graded Homework")**: Verify every migrated module against a `10,000-vector` **Production Golden I/O Replay Tape** captured from the legacy system.
- [ ] **Wire the Golden-Tape Differential Harness Directly into an Automated Repair Agent Loop**: Feed exact field-level diffs (`expected=$1.36, actual=$1.37`) back to the agent until 100% of golden vectors pass.
- [ ] **Cut Over Incrementally Using a Strangler-Fig Shadow Proxy**: Mirror live production requests to both legacy and modernized services for 14 days to confirm `99.8%+` parity before retiring the mainframe module.
