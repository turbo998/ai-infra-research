# Enterprise AI Progressive Delivery (`Canary & Shadow Rollout`), Session-Sticky Routing, Side-Effect-Free Shadow Mirroring & `SPRT` Automated Rollback Architecture Selection Guide

> **Language / 语言**: **English (Current)** | [中文](./enterprise-ai-progressive-delivery-canary-rollout-session-sticky-shadow-selection-faq.md)  
> **Scope**: End-to-end production reference architecture and sizing playbook for Cloud & AI Solution Architects, LLMOps / Platform Engineering Leads, and SREs shipping `100+` monthly prompt, LoRA, RAG, and agent releases across enterprise AI applications.  
> **Provenance**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`). Deterministic evaluation model sealed in [`experiments/enterprise_ai_progressive_delivery_canary_rollout_session_sticky_shadow_selection/results.json`](../../experiments/enterprise_ai_progressive_delivery_canary_rollout_session_sticky_shadow_selection/results.json).

![Enterprise AI Progressive Delivery & Session-Sticky Canary Summary](../../assets/svg/enterprise-ai-progressive-delivery-canary-rollout-session-sticky-shadow-selection-faq-summary.svg)

---

## ① 📌 Executive Summary (3-Minute Architect Verdict)

When enterprise engineering teams ship **`120+` monthly production AI updates** (new prompt templates, model version bumps, LoRA adapters, or MCP tool schemas across `30` applications averaging `10` turns/session), copying traditional web-service canary patterns (`Nginx / Istio per-request random 90/10 split + HTTP 500 error-rate threshold`) causes three severe production failures:
1. **Trap #1: Per-Request Random `90/10` Canary Split + `HTTP 500`-Only Monitoring (`Tier 1`, `28.4%` Mid-Session State Desync, Misses `88.0%` of Semantic Regressions — Only `12.0%` Catch Rate, `$182,000/month` Release Incident TCO)**:
   - In a 10-turn agent conversation, a stateless per-request `90/10` load balancer sends $1 - 0.9^{10} = \mathbf{65.1\%}$ **of multi-turn sessions to BOTH `v1` and `v2` mid-conversation**! Bouncing a session from `v1` on Turn 2 to `v2` on Turn 3 breaks scratchpad schemas, causes `28.4%` state desync, and destroys GPU `KV Prefix Cache` locality!
   - Worse, when a broken prompt or hallucinating model returns **`HTTP 200 OK` full of fabricated numbers or broken tool logic**, traditional Kubernetes canary controllers (`Argo Rollouts / Flagger` watching only `HTTP 5xx`) promote the bad release to `100%` GA!
2. **Trap #2: Unguarded Traffic Shadowing (`Dark Launching`) Without MCP Write-Tool Mocking (`Tier 2`, `100.0%` Duplicate Write Side-Effects — Both `v1` and `v2-Shadow` Call `create_ticket`, `send_email`, or `refund_order`! `$145,000/month` TCO)**.
3. **Trap #3: Fixed 48-Hour Manual Spot-Check Canary Soak (`Tier 3`, Blocks Release Velocity & Still Misses `27.5%` of Edge-Case Regressions, `$118,000/month` TCO)**.

Evaluated across **120 monthly production AI releases (`30` apps, `10` turns/session)**, the four progressive delivery architectures compare as follows:

| Progressive Delivery Tier | Routing, Shadowing & Rollback Gate Stack | Semantic Regression Catch | Mid-Session State Desync | Shadow Duplicate Writes | Monthly Incident TCO | Production Verdict |
|---|---|---|---|---|---|---|---|
| **🟥 Tier 1: Per-Request Random `90/10` + `HTTP 500` Gate** | Stateless per-turn random split; checks only HTTP 5xx | `12.0%` ✕ *(blind to HTTP-200 bugs)* | `28.4%` ✕ *(mid-chat version bounce)* | `0.0%` *(no shadow)* | `$182,000 / mo` ✕ *(promotes bad releases)* | ⛔ Never use stateless per-request splits or HTTP-500-only gates for AI |
| **🟨 Tier 2: Unguarded Traffic Shadowing (`Live Tools`)** | Mirrors 5% traffic to `v2` without stubbing write tools | `78.0%` | `4.5%` | **`100.0%` ✕** *(double tickets/emails!)* | `$145,000 / mo` ✕ *(corrupts prod ERP/CRM)* | ⛔ Unmocked shadow agents execute duplicate real-world mutations |
| **🟦 Tier 3: Fixed 48-Hour Manual Spot-Check Canary** | Humans manually inspect sample logs over 48 hours | `72.5%` ✕ | `2.0%` | `0.0%` | `$118,000 / mo` ✕ *(slow & incomplete)* | ⛔ Manual log review cannot scale to 120 releases/month |
| **🟩 Tier 4: `HMAC(session_id)` Sticky Canary + Write-Mocked Shadow + 4-Signal `SPRT` Auto-Rollback (`4.5m`)** | **Pins all turns of a session to 1 version; intercepts `shadow` write tools; runs Wald `SPRT` on 4 AI quality signals** | **`99.4%` 🏆** *(catches HTTP-200 drift)* | **`0.0%` 🏆** *(zero mid-session bounce)* | **`0.0%` 🏆** *(dry-run MCP stubs)* | **`$21,000 / mo` 🏆** *(-88.5% vs Tier 1)* | 🌟 **Recommended Enterprise AI Progressive Delivery Standard** |

> 💡 **Golden Rules for Enterprise AI Progressive Delivery**:
> 1. **Enforce Session-Sticky Canary Routing (`HMAC(session_id, deploy_id) % 100`)**: **"Never switch agent versions in the middle of a multi-turn conversation!"** Hash the `session_id` (or `tenant_id`) at the AI Gateway so all 10 turns of a session stay pinned to `v1` or `v2` (`0.0%` mid-session state desync, `100%` KV prefix cache locality).
> 2. **Propagate `x-traffic-mode: shadow` & Intercept Mutating MCP Tools**: When mirroring `5%` of live traffic to `v2-Shadow`, propagate `traffic.mode = shadow` in `OpenTelemetry Baggage`. Allow the shadow agent to call `READ_ONLY` search tools normally, but **intercept all `MUTATING_WRITE` tools at the MCP Gateway**—logging the shadow tool arguments for diff comparison against `v1` and returning a dry-run receipt (`0.0%` duplicate side-effects!).
> 3. **Gate Promotion on 4 AI Quality Signals via Wald's `SPRT` (`4.5-Minute` Auto-Rollback)**: Replace `HTTP 500` checks with continuous **Sequential Probability Ratio Testing (`SPRT`)** across four real-time metrics: **① Online `NLI` Groundedness Score**, **② MCP Tool Schema Error Rate**, **③ User `15s` Re-Ask / Cancel / Thumbs-Down Rate**, and **④ Tokens-per-Task Inflation** (`99.4%` regression catch in `4.5 minutes`)!

---

## ② 🎯 Customer & Architect Deep-Dive Q&A (3 The `65%` Multi-Turn Contamination Math, Shadow Write-Tool Virtualization & `4.5-Minute SPRT` Rollbacks Pain Points)

### Q1: Why does a `10%` stateless per-request canary contaminate `65.1%` of 10-turn agent sessions?
- **The Compound Probability of Stateless Per-Turn Load Balancing ($1 - 0.9^{10} = 65.1\%$)**:
  - If an ingress controller rolls a 10-sided die on every HTTP POST, the probability that a 10-turn session never hits the `10%` canary is $0.9^{10} = 34.9\%$. Thus, `65.1%` of users experience a mid-conversation version swap between `v1` and `v2`—breaking prompt prefix caches and causing `28.4%` scratchpad schema errors!
  - Hashing `HMAC(session_id, deploy_id) % 100 < 10` guarantees that **exactly `10.0%` of sessions run entirely on `v2` and `90.0%` run entirely on `v1`** (`0.0%` mid-session contamination).

### Q2: How do we shadow-test a multi-step agent on live production traffic without the shadow agent sending duplicate emails or creating duplicate Jira/ServiceNow tickets?
- **Baggage-Propagated Shadow Mode (`x-traffic-mode: shadow`) at the MCP Tool Gateway**:
  - When the AI Gateway forks a shadow request to `v2-Shadow`, it attaches `OpenTelemetry Baggage: traffic.mode=shadow`.
  - When `v2-Shadow` invokes `POST /mcp/tools/create_ticket`, the MCP Gateway sees `traffic.mode == shadow`, **suppresses the upstream ServiceNow write**, compares `v2-Shadow`'s proposed JSON arguments against `v1`'s actual arguments, and returns the cached receipt from `v1` (`0.0%` duplicate writes)!

### Q3: How can `SPRT` detect a bad prompt release and roll it back in `4.5 minutes` when traditional A/B tests take 48 hours?
- **Sequential Likelihood-Ratio Boundaries on Implicit User & Judge Signals**:
  - Fixed-horizon A/B tests wait for thousands of samples to detect a 1% lift. For canary safety, we want to catch **significant regressions** (e.g., user 15-second re-ask rate jumping from `4%` to `14%`, or online `HHEM-2.1` groundedness dropping by `8 points`) as fast as mathematically possible.
  - Wald's `SPRT` updates the log-likelihood ratio after every single completed turn, crossing the rejection boundary in **`~220 turns` (`4.5 minutes`)** and triggering an immediate automated rollback!

---

## ③ 🧭 Visual Architecture Diagrams (Decision Tree + Session-Sticky Canary & Mocked Shadow Topology + `4.5-Minute SPRT` Rollback Sequence)

### 3.1 Enterprise AI Progressive Delivery Decision Tree

```mermaid
flowchart TD
    A["Enterprise AI Progressive Delivery Selection<br/>(30 AI Apps | 120 Releases/Mo | 10-Turn Stateful Agent Sessions)"] --> B{"How are new prompts, models & agents rolled out and monitored?"}
    B -- "Per-request random 90/10 split + HTTP 500 monitor only (Tier 1)" --> C["⛔ 28.4% Mid-Session State Desync + Misses 88% HTTP-200 Bugs!<br/>Promotes hallucinating releases to 100% ($182K/mo TCO)"]
    B -- "Unguarded traffic shadowing with live write tools (Tier 2)" --> D["⛔ 100% Duplicate Tickets, Emails & Charges on Shadow Runs!<br/>Shadow agent mutates production systems ($145K/mo)"]
    B -- "Fixed 48-hour manual log spot-check canary soak (Tier 3)" --> E["⛔ 48-Hour Release Bottleneck & Misses 27.5% Regressions!<br/>Cannot scale to 120 monthly releases ($118K/mo)"]
    B -- "HMAC(session_id) Sticky Canary + Write-Mocked Shadow + 4-Signal SPRT (Tier 4)" --> F["🟩 Tier 4: Session-Sticky Canary + Mocked Shadow + 4.5m SPRT Gate 🏆<br/>(99.4% Regression Catch | 0.0% State Desync | 0.0% Shadow Dupes | -88.5% TCO)"]
    style F fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#f8fafc
    style C fill:#7f1d1d,stroke:#ef4444,color:#fecaca
    style D fill:#7f1d1d,stroke:#ef4444,color:#fecaca
    style E fill:#7f1d1d,stroke:#ef4444,color:#fecaca
```

### 3.2 Tier 4 `HMAC(session_id)` Sticky Canary, Write-Mocked Shadow & 4-Signal `SPRT` Auto-Rollback Architecture

```mermaid
flowchart LR
    subgraph Ingress["1. Session-Sticky & Shadow Router"]
        Req["💬 Multi-Turn Request<br/>(session_id)"] --> Hash["📌 HMAC(session_id, deploy_id) % 100<br/>Pins all 10 turns to 1 version<br/>(0.0% mid-chat desync!)"]
        Hash -- "90% Sticky Sessions" --> V1["🟢 v1 Production Baseline"]
        Hash -- "10% Sticky Sessions" --> V2["🟡 v2 Canary Ring"]
        Hash -. "5% Async Mirror (x-traffic-mode: shadow)" .-> V2S["👻 v2 Shadow Instance"]
    end

    subgraph MCP["2. MCP Gateway Shadow Write Firewall"]
        V1 & V2 --> Live["⚡ Normal Tool Execution"]
        V2S --> Intercept{"🛡️ Shadow Mode &<br/>Mutating Tool?"}
        Intercept -- "Read-Only Tool" --> Live
        Intercept -- "Write Tool (POST/Email/ERP)" --> Stub["🧪 Block External Write! Log Arg Diff<br/>& Return Dry-Run Stub (0% Dupes)"]
    end

    subgraph Gate["3. 4-Signal SPRT Auto-Rollback Controller (4.5 Min)"]
        V1 & V2 & V2S --> SPRT["📈 Real-Time SPRT Comparison:<br/>1. Online NLI Groundedness | 2. Tool Schema Errors<br/>3. 15s User Re-Ask Rate | 4. Tokens/Task<br/>Auto-aborts bad canary in 4.5 minutes!"]
    end
```

### 3.3 End-to-End Sequence: Shadow Write Interception Followed by `4.5-Minute SPRT` Auto-Rollback on an `HTTP 200` Regression

```mermaid
sequenceDiagram
    autonumber
    participant User as Active Users (session_id)
    participant GW as AI Gateway (Sticky Hash + Shadow)
    participant V2 as v2 Canary / Shadow Agent
    participant MCP as MCP Gateway (Write Interceptor)
    participant SPRT as 4-Signal SPRT Rollout Controller

    Note over GW,SPRT: Phase 1: 5% Write-Mocked Shadow Mirroring (`x-traffic-mode: shadow`)
    User->>GW: 1. Send support turn; primary served by `v1`, async shadow copy sent to `v2-Shadow`
    V2->>MCP: 2. `v2-Shadow` calls mutating tool `issue_credit(amount=500)`
    MCP-->>V2: 3. Detect `x-traffic-mode: shadow` -> block real credit, log argument diff & return dry-run stub (`0` duplicate credits!)
    Note over GW,SPRT: Phase 2: 10% Session-Sticky Canary (`HMAC(session_id) % 100 < 10`)
    User->>GW: 4. Route 220 sticky canary sessions to `v2` (all returning `HTTP 200 OK`)
    SPRT->>SPRT: 5. `T + 4.5m`: `SPRT` detects user 15s re-ask rate spiked (`3.8% -> 15.2%`) & NLI groundedness dropped `9.4 pts`!
    SPRT->>GW: 6. ⚡ Automatically abort `v2` rollout and shift traffic back to `100% v1` (`99.4%` catch rate)!
```

---

## ④ 📊 4-Tier Architecture Decision Matrix

| Evaluation Dimension | 🟥 Tier 1: Random Split + `HTTP 500` | 🟨 Tier 2: Live Shadow Writes | 🟦 Tier 3: 48h Manual Soak | 🟩 Tier 4: Sticky Canary + Mocked Shadow + `SPRT` |
|---|---|---|---|---|
| **Semantic Regression Catch Rate** | `12.0%` ✕ (blind to HTTP-200 bugs) | `78.0%` | `72.5%` ✕ | 🏆 **`99.4%` (4-signal `SPRT` sequential test)** |
| **Mid-Session State Desync Rate** | `28.4%` ✕ (65% sessions bounce) | `4.5%` | `2.0%` | 🏆 **`0.0%` (`HMAC(session_id)` sticky version pinning)** |
| **Shadow Duplicate Write Rate** | N/A | `100.0%` ✕ (double tickets/emails) | N/A | 🏆 **`0.0%` (`x-traffic-mode: shadow` MCP dry-run stub)** |
| **Rollback Detection Lead Time** | Missed (promoted to 100%) | ~60 minutes | `2,880 min` (48 hours) | 🏆 **`4.5 minutes` (automated `SPRT` boundary trip)** |
| **120-Release Monthly TCO** | `$182,000 / mo` ✕ | `$145,000 / mo` ✕ | `$118,000 / mo` ✕ | **`$21,000 / mo` 🏆 (-88.5% vs Tier 1)** |

---

## ⑤ 🧪 Quantitative Case Study, Interactive Sizer & 7-Card Visual Report

- **Interactive HTML Sizer**: 👉 [Open Enterprise AI Progressive Delivery & Canary Sizer](../reports/enterprise-ai-progressive-delivery-canary-rollout-session-sticky-shadow-sizer.html)
- **7-Card 4:5 Visual Benchmark Report**: 👉 [Open 7-Card Visual Benchmark Deck](../reports/enterprise-ai-progressive-delivery-canary-rollout-session-sticky-shadow-selection-cards.html)
- **Deterministic Verification**:
  ```bash
  python experiments/enterprise_ai_progressive_delivery_canary_rollout_session_sticky_shadow_selection/run.py --check
  python -m unittest tests/test_enterprise_ai_progressive_delivery_canary_rollout_session_sticky_shadow_selection.py
  ```

---

## ⑥ 🛠️ Production Readiness & Architecture Review Checklist

- [ ] **Pin Multi-Turn Canary Routing by `HMAC(session_id, deployment_id)`**: Never use stateless per-request random load balancing on multi-turn conversational or agentic workloads (`0.0%` mid-session version bounce).
- [ ] **Stamp Shadow Traffic with `OpenTelemetry Baggage: traffic.mode=shadow` and Stub Mutating MCP Tools**: Ensure shadow agents never execute real external writes (`send_email`, `create_ticket`, `charge_card`).
- [ ] **Upgrade Canary Rollout Controllers (`Argo Rollouts / Flagger`) Beyond `HTTP 500` Checks**: Gate promotion on four AI-specific golden signals (`Online NLI Groundedness`, `MCP Tool Validation Error Rate`, `User 15s Re-Ask/Cancel Rate`, and `Tokens-per-Task`).
- [ ] **Use Wald's Sequential Probability Ratio Test (`SPRT`) for Continuous Rollback Evaluation**: Abort degraded canary releases automatically in `<5 minutes` rather than waiting for a 48-hour manual soak.
- [ ] **Version Prompts, Model IDs, Chunking Configs, and Tool Schemas Together as an Immutable `Release Bundle`**: Guarantee 1-second atomic rollback across the entire AI stack.
