# Enterprise Long-Context & Multi-Turn Agent Context Compression, Prefix-Cache-Preserving History Pruning (`K=8 Epoch Compaction`) & Tool-Output Slimming Architecture Selection Guide: Why Sliding-Window FIFO Deletion Destroys `88%` of KV Prefix Cache Hits!

> **Language / 语言**: **English (Current)** | [中文](./enterprise-llm-context-compression-history-pruning-prefix-cache-budget-selection-faq.md)  
> **Scope**: End-to-end production reference architecture and sizing playbook for Cloud & AI Solution Architects, Multi-Agent Platform Leads, and AI FinOps Engineers governing long-horizon (`20–50` turn) conversational and agentic workflows.  
> **Provenance**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`). Deterministic evaluation model sealed in [`experiments/enterprise_llm_context_compression_history_pruning_prefix_cache_budget_selection/results.json`](../../experiments/enterprise_llm_context_compression_history_pruning_prefix_cache_budget_selection/results.json).

![Enterprise Multi-Turn Context Compression & Prefix-Cache Preservation Summary](../../assets/svg/enterprise-llm-context-compression-history-pruning-prefix-cache-budget-selection-faq-summary.svg)

---

## ① 📌 Executive Summary (3-Minute Architect Verdict)

In multi-turn enterprise coding assistants, SRE troubleshooting agents, and customer support workflows (**`1,500,000` monthly sessions averaging `24` turns/session**), raw context windows swell to **`52,000+ tokens` by Turn 20**—driven 75% by verbose MCP tool JSON responses, stack traces, and retrieved snippets. Three common context-management strategies backfire in production:
1. **Trap #1: Unbounded Full-History & Raw Tool JSON Append (`Tier 1`, `52,000 tokens` at Turn 20, `2,450ms` TTFT P99, `$240,000/month` Token & GPU KV Cache TCO)**:
   - A 10,000-token REST API dump returned on Turn 3 gets re-sent and re-billed across Turns `4..24` (**paid for 21 times**)!
2. **Trap #2: Per-Turn Sliding-Window FIFO Deletion ("Keep Last 10 Turns, Drop Turn 1", `Tier 2`, Collapses KV Prefix Cache Hit Rate from `78%` Down to `12.0%` AND Drops Turn-1 Constraint Retention to `66.0%`, `$165,000/month` TCO)**:
   - **The #1 Hidden Prompt-Caching Killer**: Modern inference engines (`vLLM Automatic Prefix Caching`, `SGLang RadixAttention`, `Anthropic / OpenAI 90%-off Prompt Caching`) require **exact token-offset prefix identity** to reuse KV cache blocks.
   - When you drop `Turn_1` (`800 tokens`) on Turn 11, every subsequent token in `Turn_2..Turn_10` shifts left by 800 positions—**invalidating `100%` of the conversation history's KV cache blocks on EVERY single turn**!
3. **Trap #3: Re-Summarizing Conversation History at the Top of the Prompt on Every Turn (`Tier 3`, `18.0%` KV Cache Hit Rate, `+1,400ms` Summarization Lag, `$148,000/month` TCO)**: Mutating the rolling summary text at the front of the prompt on every turn also invalidates the KV cache for all subsequent turns!

Evaluated across **1,500,000 monthly multi-turn sessions (`24` turns/session)**, the four context-management architectures compare as follows:

| Context Management Tier | Pruning, Compaction & Tool-Output Stack | Turn-20 Constraint Retention | `KV` Prefix Cache Hit Rate | Turn-20 `TTFT P99` | 1.5M-Session Monthly TCO | Production Verdict |
|---|---|---|---|---|---|---|---|
| **🟥 Tier 1: Unbounded Full-History & Raw Tool Append** | Appends all raw tool JSONs & turns until limit | `81.5%` *(lost-in-the-middle)* | `78.0%` | `2,450.0 ms` ✕ *(52K prefill lag)* | `$240,000 / mo` ✕ *(21x dead JSON tax)* | ⛔ Context bloat explodes latency and GPU KV memory |
| **🟨 Tier 2: Per-Turn Sliding-Window `FIFO` Drop** | Drops oldest turn on every new turn (`[T2..T11]`) | `66.0%` ✕ *(forgets Turn-1 rules)* | **`12.0%` ✕** *(offset shift kills cache!)* | `1,180.0 ms` | `$165,000 / mo` ✕ *(loses 90% cache discount)* | ⛔ Never shift token offsets on every turn when using Prefix Caching |
| **🟦 Tier 3: Per-Turn Full Rolling Re-Summarization** | Rewrites top summary block after every turn | `88.0%` | **`18.0%` ✕** *(mutated prefix)* | `1,650.0 ms` ✕ | `$148,000 / mo` ✕ | ⛔ Mutating the summary prefix every turn invalidates downstream KV blocks |
| **🟩 Tier 4: `K=8` Epoch Block Compaction + Tool `JQ` Slimming (`artifact://`) + Pinned Goal Ledger** | **Append-only for 8 consecutive turns (`89.5%` cache hit); compacts only at epoch boundary + slims tool JSONs by `-78%`** | **`98.4%` 🏆** *(pinned 250t ledger)* | **`89.5%` 🏆** *(+77.5 pts vs FIFO)* | **`340.0 ms` 🏆** *(7.2x faster!)* | **`$26,500 / mo` 🏆** *(-89.0% vs Tier 1)* | 🌟 **Recommended Enterprise Multi-Turn Context Standard** |

> 💡 **Golden Rules for Enterprise Multi-Turn Context Compression**:
> 1. **Never Shift Token Offsets on Every Turn — Use Sawtooth `K=8` Epoch Block Compaction**: Keep the prefix `[System + Pinned_Ledger + Epoch_Summaries + Active_Turns_1..7]` **100% immutable and append-only across 8 consecutive turns** so `89.5%` of input tokens hit the GPU KV Prefix Cache (`90%` cost discount, zero prefill recompute). Compact older turns into a new immutable `[Epoch_N_Summary]` block **only once every `K=8` turns** (or when the active tail exceeds `16,000 tokens`)!
> 2. **Immediate Tool-Output Slimming & `artifact://` Pointer Offload (`-78%` Tool Tokens)**: Intercept verbose MCP tool outputs (`>1,000 tokens`), persist the full payload in Redis/S3 under `artifact://call_92`, and inject only a **`450-token` JQ/Schema projection** into the chat history—replacing completed tool outputs with `40-token` tombstones at the next epoch boundary!
> 3. **Pin a `250-Token` Structured Goal & Constraint Ledger**: Store Turn-1 user hard constraints, verified entity IDs, and completed milestones in a dedicated JSON block right below the system prompt (`98.4%` Turn-20 constraint retention)!

---

## ② 🎯 Customer & Architect Deep-Dive Q&A (3 Why FIFO Windows Break PagedAttention, Slimming 12K Tool JSONs & Long-Horizon Amnesia Pain Points)

### Q1: Why does dropping the oldest message on every turn (`Tier 2` Sliding Window) destroy `88%` of GPU KV Prefix Cache hits?
- **Token Offset Shifts Invalidate Every Downstream `PagedAttention` Block Hash**:
  - In `vLLM`, `SGLang`, and cloud Prompt Caching, KV cache blocks are keyed by the **cumulative token sequence hash from index `0`** (and incorporate RoPE positional embeddings at exact index $i$).
  - When Turn 11 deletes `Turn_1` (`800 tokens`) from the front of the list, `Turn_2..Turn_10` shift from positions `[1000..15000]` to `[200..14200]`—causing a **100% cache miss across all 14,000 historical tokens on every turn**!
  - By keeping the history strictly **append-only for `K=8` turns** and compacting in discrete epoch steps (a sawtooth pattern), 7 out of every 8 turns enjoy a `100%` prefix cache hit (`89.5%` overall cache hit rate)!

### Q2: How do we stop a `12,000-token` JSON response or log dump from Turn 3 from being billed 20 times across Turns `4..24`?
- **Two-Stage Tool Output Slimming (`JQ Projection` + `Epoch Tombstoning`)**:
  - At Turn 3, the MCP Gateway saves the raw 12,000-token JSON to `artifact://call_92` and injects only the `450-token` relevant slice (plus a `read_artifact("artifact://call_92")` handle if the agent needs to inspect more fields).
  - When the `K=8` epoch compaction runs at Turn 8, that tool output is collapsed into a `40-token` tombstone (`[Tool get_order_logs -> found timeout on shard-3; raw at artifact://call_92]`), cutting tool token bloat by **`78.0%`**!

### Q3: How does Tier 4 achieve `98.4%` retention of Turn-1 user constraints at Turn 20 when standard histories drift below `82%`?
- **The Pinned Structured Goal & Constraint Ledger (`250 Tokens`)**:
  - Instead of leaving critical constraints (*"Must use Python 3.9, no external packages, customer_id=C-901"*) buried in natural-language chat turns where they get diluted or summarized away, a lightweight state extractor maintains a **pinned 250-token JSON ledger** refreshed at each `K=8` epoch boundary!

---

## ③ 🧭 Visual Architecture Diagrams (Decision Tree + Sawtooth `K=8` Epoch Compaction Layout + Execution Sequence)

### 3.1 Enterprise Multi-Turn Context Compression Decision Tree

```mermaid
flowchart TD
    A["Enterprise Multi-Turn Context Compression Selection<br/>(1.5M Sessions/Mo | 24 Turns/Session | 52,000 Raw Tokens by Turn 20)"] --> B{"How is conversation & tool history managed across 24+ turns?"}
    B -- "Unbounded full-history & raw tool JSON append (Tier 1)" --> C["⛔ 2,450ms Turn-20 TTFT & $240K/mo TCO!<br/>Dead 10K-token tool JSONs re-billed 20 times over"]
    B -- "Per-turn sliding-window FIFO drop of oldest turn (Tier 2)" --> D["⛔ KV Prefix Cache Hit Rate Drops to 12.0%!<br/>Token offset shift invalidates cache every turn & loses 34% constraints"]
    B -- "Re-summarize entire history at top of prompt every turn (Tier 3)" --> E["⛔ 18.0% KV Cache Hit Rate & +1.4s Lag!<br/>Mutating top summary invalidates downstream KV blocks ($148K/mo)"]
    B -- "K=8 Epoch Block Compaction + Tool JQ Slimming + Pinned Ledger (Tier 4)" --> F["🟩 Tier 4: K=8 Epoch Compaction + Artifact Offload + Pinned Ledger 🏆<br/>(89.5% KV Cache Hit | 98.4% Constraint Memory | 340ms TTFT | -89.0% TCO)"]
    style F fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#f8fafc
    style C fill:#7f1d1d,stroke:#ef4444,color:#fecaca
    style D fill:#7f1d1d,stroke:#ef4444,color:#fecaca
    style E fill:#7f1d1d,stroke:#ef4444,color:#fecaca
```

### 3.2 Tier 4 Prefix-Cache-Preserving Prompt Layout (`K=8` Epoch Compaction + `artifact://` Tool Slimming)

```mermaid
flowchart TB
    subgraph Cached["1. 100% Immutable Prefix Across Turns 9..15 (89.5% GPU KV Cache Hit!)"]
        Sys["📌 Block 0: Static System Prompt & MCP Schemas"]
        Ledger["📋 Block 1: Pinned Goal & Constraint Ledger (250 Tokens)<br/>{hard_constraints: [...], verified_ids: {...}} -> 98.4% Retention!"]
        Epoch1["📦 Block 2: Immutable Compacted Block [Epoch_1: Turns 1..8]"]
        Sys --> Ledger --> Epoch1
    end

    subgraph Active["2. Append-Only Active Tail (Turns 9..15: Zero Token-Offset Shift!)"]
        Epoch1 --> T9["💬 Turns 9..14: Conversation & Slimmed Tool Outputs (Cached!)"]
        T9 --> T15["⚡ Turn 15 (Current): Prefills only new 600-token delta (340ms TTFT!)"]
    end

    subgraph Slim["3. Tool-Output Slimmer & Artifact Pointer Offload (-78% Tool Tokens)"]
        Raw["🐘 Raw MCP Tool JSON (12,000 Tokens)"] --> Blob[("🗄️ Redis/S3 artifact://call_92")]
        Raw --> Proj["✂️ 450-Token Schema Projection"]
        Proj --> T15
    end
```

### 3.3 End-to-End Sequence: 8 Consecutive Cache-Hit Turns Followed by Asynchronous Epoch Block Compaction

```mermaid
sequenceDiagram
    autonumber
    participant User as User / Agent Loop
    participant Mgr as Context Manager (K=8 Epoch + Tool Slimmer)
    participant Blob as Artifact Store (Redis / S3)
    participant LLM as LLM Engine (vLLM APC / Prompt Cache)

    User->>Mgr: 1. Turns 9..15: MCP tool returns `12,000-token` raw JSON payload
    Mgr->>Blob: 2. Offload raw JSON to `artifact://call_92`; project `450-token` summary into Turn 15
    Mgr->>LLM: 3. Send prompt: Prefix `[Sys + Ledger + Epoch_1 + Turns_9..14]` is 100% byte-identical to Turn 14!
    LLM-->>User: 4. Hit GPU KV Prefix Cache (`89.5%` hit rate) -> compute only 600 new tokens (`340ms` TTFT)!
    Note over Mgr,LLM: End of Turn 16 (Active tail reaches K=8 turns): Trigger Epoch Compaction
    Mgr->>Mgr: 5. Compact `Turns 9..14` into immutable `[Epoch_2_Summary]`, replace old tool outputs with 40-tok tombstones & refresh `Pinned Ledger`
    Mgr->>LLM: 6. Turns 17..24 resume zero-shift append-only KV cache hits on top of `[Epoch_1 + Epoch_2]`!
```

---

## ④ 📊 4-Tier Architecture Decision Matrix

| Evaluation Dimension | 🟥 Tier 1: Unbounded Append | 🟨 Tier 2: Per-Turn `FIFO` Window | 🟦 Tier 3: Per-Turn Re-Summary | 🟩 Tier 4: `K=8` Epoch + Tool Slimming + Ledger |
|---|---|---|---|---|
| **Turn-20 Constraint Retention** | `81.5%` (middle dilution) | `66.0%` ✕ (dropped Turn 1) | `88.0%` | 🏆 **`98.4%` (`250-token` pinned JSON constraint ledger)** |
| **`KV` Prefix Cache Hit Rate** | `78.0%` | `12.0%` ✕ (offset shift miss!) | `18.0%` ✕ (prefix mutated) | 🏆 **`89.5%` (strictly append-only across 8-turn epochs)** |
| **Turn-20 `TTFT P99` Latency** | `2,450.0 ms` ✕ (52K prefill) | `1,180.0 ms` | `1,650.0 ms` ✕ | 🏆 **`340.0 ms` (7.2x faster than Tier 1)** |
| **Tool JSON Bloat Reduction** | `0.0%` (paid 21 times) | Truncates blindly | Drops exact IDs | 🏆 **`-78.0%` (`JQ` projection + `artifact://` pointer offload)** |
| **1.5M-Session Monthly TCO** | `$240,000 / mo` ✕ | `$165,000 / mo` ✕ | `$148,000 / mo` ✕ | **`$26,500 / mo` 🏆 (-89.0% vs Tier 1)** |

---

## ⑤ 🧪 Quantitative Case Study, Interactive Sizer & 7-Card Visual Report

- **Interactive HTML Sizer**: 👉 [Open Enterprise Multi-Turn Context Compression & Prefix-Cache Sizer](../reports/enterprise-llm-context-compression-history-pruning-prefix-cache-budget-sizer.html)
- **7-Card 4:5 Visual Benchmark Report**: 👉 [Open 7-Card Visual Benchmark Deck](../reports/enterprise-llm-context-compression-history-pruning-prefix-cache-budget-selection-cards.html)
- **Deterministic Verification**:
  ```bash
  python experiments/enterprise_llm_context_compression_history_pruning_prefix_cache_budget_selection/run.py --check
  python -m unittest tests/test_enterprise_llm_context_compression_history_pruning_prefix_cache_budget_selection.py
  ```

---

## ⑥ 🛠️ Production Readiness & Architecture Review Checklist

- [ ] **Replace Per-Turn `FIFO` Sliding Windows with `K=8` Epoch Block Compaction**: Never delete or shift early turns on every request; keep the prefix strictly append-only within each 8-turn epoch to preserve `89.5%` KV cache hits.
- [ ] **Never Mutate Top-of-Prompt Summaries on Every Turn**: Update compacted epoch blocks and pinned ledgers only at epoch boundaries (`every 8 turns` or `>16K` active tail tokens).
- [ ] **Slim Verbose MCP Tool Outputs Immediately via Schema Projection & `artifact://` Pointers**: Store payloads `>1,000 tokens` in Redis/S3 and inject only a `450-token` summary plus an on-demand read handle.
- [ ] **Collapse Completed Tool Outputs into `40-Token` Tombstones During Epoch Compaction**: Stop paying 20 times for dead stack traces and JSON dumps from earlier turns.
- [ ] **Maintain a `250-Token` Pinned Goal & Constraint Ledger**: Guarantee `98.4%` retention of initial user requirements, target versions, and entity IDs across 40+ turn sessions.
