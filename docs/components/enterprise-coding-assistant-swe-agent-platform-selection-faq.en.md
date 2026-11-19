# Enterprise AI Coding Assistant & Autonomous `SWE Agent` Platform Architecture Selection Guide: Un-Indexed Web Chat vs. Single-File `FIM` Only vs. Un-Gated Agents vs. Dual-Engine `125ms FIM` + `Tree-sitter/SCIP` Symbol Graph + Ephemeral `CI/SAST` Self-Healing Sandbox

> **Language / 语言**: **English (Current)** | [中文](./enterprise-coding-assistant-swe-agent-platform-selection-faq.md)  
> **Scope**: End-to-end production reference architecture and developer-productivity / AppSec guide for Cloud & AI Solution Architects, VPs of Engineering, Developer Experience (`DevEx`) Leads, and Application Security Architects.  
> **Provenance**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`). Deterministic evaluation model sealed in [`experiments/enterprise_coding_assistant_swe_agent_platform_selection/results.json`](../../experiments/enterprise_coding_assistant_swe_agent_platform_selection/results.json).

![Enterprise AI Coding Assistant & Autonomous SWE Agent Selection Summary](../../assets/svg/enterprise-coding-assistant-swe-agent-platform-selection-faq-summary.svg)

---

## ① 📌 Executive Summary (3-Minute Architect Verdict)

When rolling out AI Coding Assistants and Autonomous Software Engineering (`SWE`) Agents across **`2,000` software engineers and `4,500` internal repositories (including a `15-Million-LOC` core monorepo)**, many organizations discover a painful paradox: **junior developers write code 20% faster, but senior engineers spend twice as long reviewing broken or insecure AI-generated Pull Requests!** Three architectural mistakes cause this bottleneck:
1. **Mistake #1: Missing Compiler-Grade Repository Symbol Indexing (`AST / SCIP`), Causing `21.5%–38.0%` Internal API Hallucinations (Tiers 1–2)**: Without a cross-repository symbol call graph and internal SDK documentation index, the LLM guesses proprietary method signatures and hallucinates non-existent package names (exposing the build pipeline to **"Slopsquatting"** supply-chain attacks);
2. **Mistake #2: Routing Keystroke Inline Autocomplete (`FIM`) to a Heavy Frontier Reasoning Model (`890ms` P95 Latency, Tier 3)**: Inline ghost-text completion must arrive in `< 150ms`; waiting `890ms` for a 70B/frontier model means the developer has already typed the next word, wasting 70% of generated tokens;
3. **Mistake #3: Letting Autonomous Coding Agents Open PRs Without an Automated `Compiler + Unit Test + SAST` Verification Loop (`7.2%` Vulnerability Escape Rate, Tier 3)**: Dumping un-compiled diffs containing SQLi/SSRF or hardcoded secrets onto human reviewers.

Evaluated across **2,000 engineers and 4,500 internal repositories (`15M` LOC)**, the four AI coding platform architectures compare as follows:

| Architectural Tier | Core AI Coding & SWE Stack | Multi-File PR Resolution | Inline `FIM` P95 Latency | Vulnerability / Secret Escape | Internal API Hallucination | Monthly Net Platform + Review TCO | Production Verdict |
|---|---|---|---|---|---|---|---|
| **🟥 Tier 1: Un-Indexed Generic Web Chat** | Manual copy-paste into web LLM without repo context | `18.5%` ✕ *(blind to repo graph)* | `2,150 ms` ✕ | `16.4%` ✕ *(unscanned snippets)* | `38.0%` ✕ *(invents SDK calls)* | `$190,000 / mo` ✕ *(rework burden)* | ⛔ High IP leak risk & severe API hallucinations |
| **🟨 Tier 2: Single-File IDE `FIM` Only** | Fast single-file tab autocomplete without multi-file agent | `34.0%` ✕ *(single-tab scope)* | `145 ms` | `9.8%` ✕ | `21.5%` ✕ | `$125,000 / mo` | ⚠️ Helps with boilerplate, fails on cross-repo tasks |
| **🟦 Tier 3: Un-Gated Frontier Model on Raw Grep** | Uses frontier model for both FIM & edits without SCIP/SAST | `62.4%` | `890 ms` ✕ *(typing lag)* | `7.2%` ✕ *(burdens reviewers)* | `8.4%` | `$148,000 / mo` ✕ | ⚠️ Sluggish autocomplete & unverified PR bloat |
| **🟩 Tier 4: Dual-Engine (`125ms FIM` + `SCIP` Frontier Agent) + `CI/SAST` Self-Healing Loop** | **`7B/14B FIM` (`125ms`) + `Tree-sitter/SCIP` Symbol Graph + Ephemeral `LSP + Test + CodeQL/Semgrep + Artifactory` Loop** | **`84.6%` 🏆** *(+50.6 pts vs Tier 2)* | **`125 ms` 🏆** *(instant ghost text)* | **`0.0%` 🏆** *(pre-PR SAST gate)* | **`0.6%` 🏆** *(exact symbol graph)* | **`$52,000 / mo` 🏆** *(-72.6%)* | 🌟 **Recommended Enterprise AI SWE Platform Standard** |

> 💡 **Golden Rule for Enterprise AI Coding & SWE Platform Architects**: **"Build on three pillars: (1) decouple your `Inline FIM Engine` (`7B–14B Coder FP8` with 150ms keystroke debounce and speculative decoding for `125ms` P95 ghost text) from your `Multi-File SWE Agent Engine` (Frontier Reasoning Model); (2) index internal repositories using `Tree-sitter AST` syntax boundaries and `SCIP / LSIF` compiler symbol graphs (`0.6%` internal API hallucination); and (3) require every autonomous Agent diff to pass an ephemeral devcontainer self-healing loop (`LSP Compiler -> Unit Tests -> CodeQL/Semgrep SAST -> Artifactory Lockfile Verification`) before opening a Pull Request."**

---

## ② 🎯 Customer & Architect Deep-Dive Q&A (3 FIM Latency, Internal API Hallucination & Unverified PR Pain Points)

### Q1: Why must IDE keystroke autocomplete (`FIM`) and multi-file autonomous coding (`SWE Agent`) run on separate model tiers?
- **The `150ms` Human Typing Flow Boundary**:
  - Keystroke Fill-in-the-Middle (`<|fim_prefix|>...<|fim_suffix|>...<|fim_middle|>`) fires hundreds of times per hour per developer and generates 15–40 tokens. Serving it from a dedicated **`7B–14B Coder FP8` model** with a `150ms` client debounce achieves **`125ms` P95 latency** at a fraction of frontier token cost.
  - Complex multi-file bug fixes, migrations, and unit-test generation are routed to a **Frontier Reasoning Agent** backed by `82%` Prefix KV Caching.

### Q2: How do `Tree-sitter AST` + `SCIP` Symbol Graphs and Artifactory Lockfile Gating eliminate internal SDK hallucinations (`0.6%`) and "Slopsquatting" attacks (`0.0%`)?
- **Compiler-Grade Call Graphs Over Naive Text Chunking**:
  - Naive character splitters chop functions in half and lose cross-repository call edges. **`Tree-sitter`** chunks code strictly along AST class/function boundaries, while **`SCIP` (Source Code Index Protocol)** provides exact `Go-to-Definition` and `Find-All-References` edges across all 4,500 repositories.
  - Inside the agent's build sandbox, public package registries (`pypi.org`, `npmjs.com`) are blocked; all dependencies must resolve against the enterprise **Artifactory / Nexus allowlist**, neutralizing hallucinated "slopsquatting" package names.

### Q3: How does the Ephemeral Devcontainer Self-Healing Loop prevent AI-generated PRs from overwhelming senior code reviewers?
- **Mandatory Pre-PR Verification (`Up to 3 Self-Healing Iterations`)**:
  - Before a human is ever notified, the agent applies its patch inside an isolated container and runs: (1) `LSP / Compiler` type checking, (2) affected unit tests (`pytest / jest / go test`), and (3) `CodeQL / Semgrep` SAST + `Gitleaks` secret scanning. Any compiler failure, failing test, or SAST finding is fed back into the agent to fix autonomously (`84.6%` PR resolution, `0.0%` vulnerability escape).

---

## ③ 🧭 Visual Architecture Diagrams (Decision Tree + Dual-Engine SWE Topology + Sequence Flow)

### 3.1 Enterprise AI Coding Assistant & SWE Agent Selection Decision Tree

```mermaid
flowchart TD
    A["Enterprise AI Coding & SWE Agent Platform Selection<br/>(2,000 Engineers + 4,500 Internal Repos + 15M-LOC Monorepo)"] --> B{"Repository data residency classification?"}
    B -- "85% Standard Enterprise Product Repos" --> C["☁️ Managed Cloud Enterprise (e.g., GitHub Copilot Enterprise)<br/>(Custom Repo Indexing + Zero Data Retention + IP Indemnity)"]
    B -- "15% Air-Gapped Sovereign / Defense Enclaves" --> D["🏠 Self-Hosted Pool (vLLM Qwen2.5-Coder-32B/7B + Continue/Tabby)"]
    C & D --> E{"Enforces Dual-Engine FIM/Agent Split, Tree-sitter/SCIP Graph<br/>& Ephemeral LSP/Test/SAST Self-Healing Sandbox?"}
    E -- "No (Single slow model or unverified PR dumps, Tiers 1–3)" --> F["⛔ 890ms Typing Lag + 38% API Hallucinations + 7.2% Vuln Escape!"]
    E -- "Yes (125ms FIM + SCIP Graph + CI/SAST Sandbox Loop, Tier 4)" --> G["🟩 Tier 4: Dual-Engine FIM + SCIP Repo Agent + SAST Self-Healing Loop 🏆<br/>(84.6% Multi-File PR Resolution | 125ms FIM | 0.0% Vuln Escape | -72.6% TCO)"]
    style G fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#f8fafc
    style F fill:#7f1d1d,stroke:#ef4444,color:#fecaca
```

### 3.2 Tier 4 Dual-Engine (`125ms FIM` + `SCIP` Symbol Graph `SWE Agent`) + Self-Healing Sandbox Architecture

```mermaid
flowchart LR
    subgraph IDE["1. Developer IDE & Issue Triggers"]
        Dev["👩‍💻 2,000 Engineers (IDE / PR / Issue)"] -->|Lane A: 150ms Keystroke Debounce| FIM["⚡ Dedicated 7B/14B FIM Engine<br/>(Speculative Decoding | 125ms P95)<br/>Instant Ghost-Text Autocomplete"]
        Dev -->|Lane B: Multi-File Refactor / Issue| Agent["🧠 Autonomous SWE Agent<br/>(Frontier Reasoning Model)"]
    end

    subgraph Context["2. Compiler-Grade Code Intelligence"]
        Agent <--> SCIP["🕸️ Tree-sitter AST + SCIP Call Graph<br/>+ Internal SDK / Protobuf / OpenAPI MCP<br/>(0.6% Internal API Hallucination)"]
    end

    subgraph Sandbox["3. Ephemeral Self-Healing Verification Sandbox"]
        Agent --> Box["🛡️ Disposable Devcontainer Sandbox<br/>1. LSP Type & Compiler Check<br/>2. Unit Tests + Artifactory Lockfile Gate<br/>3. CodeQL / Semgrep SAST + Gitleaks"]
        Box -->|Opens PR Only When 100% Green!| PR["✅ Verified Pull Request Ready for Human Review"]
    end
```

### 3.3 End-to-End Sequence: Refactoring an Internal API Across 18 Call Sites with Self-Healing SAST Verification

```mermaid
sequenceDiagram
    autonumber
    participant Lead as Staff Engineer
    participant Agent as Autonomous SWE Agent
    participant SCIP as SCIP Symbol Graph & Internal Docs MCP
    participant Box as Ephemeral CI + SAST Sandbox
    participant Git as Enterprise Git Repo (Pull Request)

    Lead->>Agent: Assign task: "Migrate `AuthClient.verify()` to v2 signature across all callers"
    Agent->>SCIP: Query `SCIP` call graph for all references to `AuthClient.verify` + v2 Protobuf spec
    SCIP-->>Agent: Return exact file paths, line numbers & type signatures across 18 call sites (0.6% hallucination!)
    Agent->>Box: Apply multi-file patch in isolated sandbox & trigger Iteration #1 verification
    Box-->>Agent: Unit tests pass, but `Semgrep SAST` flags an unredacted token log & an unapproved external package!
    Agent->>Box: Iteration #2 self-heal: scrub token log, switch to Artifactory-approved internal package, re-run `LSP + Pytest + Semgrep`
    Box-->>Agent: All 4 gates (Compile + Test + SAST + Lockfile) pass 100%!
    Agent->>Git: Open verified Pull Request with attached test & SAST attestation (cuts human review time by 75%)!
```

---

## ④ 📊 4-Tier Architecture Decision Matrix

| Evaluation Dimension | 🟥 Tier 1: Un-Indexed Web Chat | 🟨 Tier 2: Single-File `FIM` Only | 🟦 Tier 3: Un-Gated Frontier Agent | 🟩 Tier 4: Dual-Engine + `SCIP` + `SAST` Sandbox |
|---|---|---|---|---|
| **Multi-File PR Resolution Rate** | `18.5%` ✕ | `34.0%` ✕ | `62.4%` | 🏆 **`84.6%` (`SCIP` call graph + 3-turn sandbox self-heal)** |
| **IDE Inline `FIM` P95 Latency** | `2,150 ms` ✕ | `145 ms` | `890 ms` ✕ (too slow for typing) | 🏆 **`125 ms` (dedicated `7B/14B FIM` + speculative decode)** |
| **Vulnerability / Secret Escape** | `16.4%` ✕ | `9.8%` ✕ | `7.2%` ✕ | 🏆 **`0.0%` (mandatory pre-PR `CodeQL/Semgrep + Gitleaks`)** |
| **Internal API Hallucination Rate** | `38.0%` ✕ | `21.5%` ✕ | `8.4%` | 🏆 **`0.6%` (`Tree-sitter AST + SCIP` + Artifactory gate)** |
| **2,000-Dev Monthly Net TCO** | `$190,000 / mo` | `$125,000 / mo` | `$148,000 / mo` | **`$52,000 / mo` 🏆 (-72.6%, saves $1.65M/yr)** |

---

## ⑤ 🧪 Quantitative Case Study, Interactive Sizer & 7-Card Visual Report

- **Interactive HTML Sizer**: 👉 [Open Enterprise AI Coding Assistant & Autonomous SWE Agent Sizer](../reports/enterprise-coding-assistant-swe-agent-platform-sizer.html)
- **7-Card 4:5 Visual Benchmark Report**: 👉 [Open 7-Card Visual Benchmark Deck](../reports/enterprise-coding-assistant-swe-agent-platform-selection-cards.html)
- **Deterministic Verification**:
  ```bash
  python experiments/enterprise_coding_assistant_swe_agent_platform_selection/run.py --check
  python -m unittest tests/test_enterprise_coding_assistant_swe_agent_platform_selection.py
  ```

---

## ⑥ 🛠️ Production Readiness & Architecture Review Checklist

- [ ] **Decouple Keystroke Inline Autocomplete (`7B/14B FIM` at `< 150ms` P95) from Multi-File Reasoning (`Frontier SWE Agent`)**: Configure a `150ms` IDE typing debounce to prevent wasted autocomplete calls.
- [ ] **Index Internal Repositories with `Tree-sitter AST` Chunking and `SCIP / LSIF` Symbol Call Graphs**: Expose internal Protobuf/OpenAPI specs and SDK documentation via an internal `MCP` server.
- [ ] **Gate All Autonomous Agent Pull Requests Behind an Ephemeral Sandbox Loop (`LSP Compile -> Unit Tests -> CodeQL/Semgrep SAST -> Gitleaks`)**: Allow up to 3 automated self-healing iterations before notifying human reviewers.
- [ ] **Block Direct Public `PyPI / npm` Access Inside Agent Build Sandboxes and Enforce Enterprise `Artifactory / Nexus` Lockfile Verification to Eliminate Slopsquatting**.
