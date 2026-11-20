# High-Concurrency Enterprise Customer Service & Agent-Assist Copilot (`10,000+` Concurrent Sessions / `250 QPS`) End-to-End Architecture, `35/45/20` Tiered Offload Funnel & GPU/PTU Capacity Sizing Guide: How to Cut `3.4s` Waterfall Latency to `390ms` and Shrink the Fleet from `64` GPUs to `14`?

> **Language / 语言**: **English (Current)** | [中文](./enterprise-multimodal-customer-service-copilot-architecture-sizing-selection-faq.md)  
> **Scope**: End-to-end production reference architecture and capacity sizer guide for Cloud & AI Solution Architects, Contact Center / CX Technology VPs, and AI Infrastructure Planners sizing high-concurrency customer bots and human-agent whisper copilots.  
> **Provenance**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`). Deterministic evaluation model sealed in [`experiments/enterprise_multimodal_customer_service_copilot_architecture_sizing_selection/results.json`](../../experiments/enterprise_multimodal_customer_service_copilot_architecture_sizing_selection/results.json).

![10,000-Concurrency Customer Service & Agent-Assist Copilot Sizing Summary](../../assets/svg/enterprise-multimodal-customer-service-copilot-architecture-sizing-selection-faq-summary.svg)

---

## ① 📌 Executive Summary (3-Minute Architect Verdict)

When designing an omnichannel customer service bot and human-agent assist Copilot for **10,000 concurrent active sessions (`250 QPS` peak at 1 turn every 40 seconds = `60M turns/month`)**, enterprise architects frequently get trapped between multi-second latency freezes and million-dollar GPU bills:
1. **Trap #1: The 5-Step Serial Waterfall on a Monolithic `70B / GPT-4o` Model (Tier 1, `3,420ms` P99 TTFT, `64 H100 GPUs`, `$1,104,000/month`)**: Executing `Input Guardrail (150ms) -> Intent LLM (350ms) -> Hybrid RAG + Rerank (280ms) -> 70B LLM TTFT (1,800ms) -> Blocking Output Judge (840ms)` sequentially forces customers to stare at a spinner for 3.4 seconds while burning frontier GPU capacity on repetitive "Where is my order?" queries!
2. **Trap #2: Routing 100% of Traffic to an `8B` Small Language Model (Tier 2, `63.5%` First-Contact Resolution `FCR`)**: While an `8B SLM` is fast (`480ms`) and requires only 12 GPUs, it fails on multi-policy refund disputes and edge-case troubleshooting—triggering a 36.5% human escalation surge that costs `$274K/month` in extra contact-center agent labor;
3. **Trap #3: Serial `8B/70B` Routing Without Parallel Speculative Prefetching (Tier 3, `1,180ms` P99 TTFT)**: Running the intent router, RAG retrieval, and guardrails one after another still wastes `790ms` before the LLM even starts generating.

Evaluated across **10,000 concurrent customer and agent-assist sessions (`250 QPS` peak, `60M turns/month`)**, the four copilot architectures compare as follows:

| Architectural Tier | Pipeline & Traffic Funnel Design | End-to-End `P99 TTFT` | First-Contact Resolution (`FCR`) | Cost / 1,000 Turns | Required `GPU/PTU` Fleet | 60M-Turn Monthly TCO | Production Verdict |
|---|---|---|---|---|---|---|---|
| **🟥 Tier 1: 5-Step Serial Waterfall (`100% 70B/GPT-4o`)** | Sequential guardrails, intent, RAG, and 70B generation | `3,420.0 ms` ✕ *(severe lag)* | `81.0%` | `$18.40 / 1K` ✕ | **`64 GPUs`** ✕ | **`$1,104,000 / mo` ✕** | ⛔ Unacceptable 3.4s lag & 4.5x GPU over-provisioning |
| **🟨 Tier 2: 100% `8B SLM` Only (No Frontier Escalation)** | Routes all queries & complex disputes to an 8B model | `480.0 ms` | `63.5%` ✕ *(high escalations)* | `$1.85 / 1K` | `12 GPUs` | `$385,000 / mo` *(incl. labor)* | ⛔ Cheap GPUs, but fails complex multi-policy disputes |
| **🟦 Tier 3: Serial `8B/70B` Router + Blocking RAG** | Runs router (180ms) -> RAG (190ms) -> LLM sequentially | `1,180.0 ms` ✕ *(790ms pre/post)* | `88.5%` | `$5.20 / 1K` | `28 GPUs` | `$312,000 / mo` | ⚠️ Sequential pre-LLM steps break sub-500ms SLA |
| **🟩 Tier 4: `35/45/20` Funnel + `48ms` Parallel Speculative Fan-Out + Streaming Gate** | **`35% Redis Template Cache -> 45% 8B SLM -> 20% 70B/PTU` + Parallel `Shield + Router + RAG` (`48ms`)** | **`390.0 ms` 🏆** *(-88.6% vs Tier 1)* | **`93.8%` 🏆** *(+ Whisper Assist)* | **`$1.60 / 1K` 🏆** | **`14 GPUs` 🏆** *(-78.1% vs Tier 1)* | **`$96,000 / mo` 🏆** *(-91.3% vs Tier 1)* | 🌟 **Recommended 10K-Concurrency Copilot Standard** |

> 💡 **Golden Rule for High-Concurrency Copilot Architects**: **"Never run guardrails, intent routing, and RAG retrieval sequentially, and never size 100% of peak QPS for a 70B/Frontier model! Fire `Input Shield (18ms)`, `INT8 Router (12ms)`, and `Speculative Hybrid RAG + TEI Rerank (48ms)` concurrently in a single `48ms` parallel fan-out, then route traffic through a `35% Redis Template Cache + 45% 8B SLM + 20% 70B/Frontier` funnel—hitting `390ms` P99 TTFT and `93.8%` FCR on just `14 GPUs` (`-91.3%` TCO)."**

---

## ② 🎯 Customer & Architect Deep-Dive Q&A (3 Whiteboard Capacity Sizer, Parallel Fan-Out & Cache Privacy Pain Points)

### Q1: How do Solution Architects calculate the exact GPU / PTU fleet size for `10,000` concurrent users on a whiteboard in 3 minutes?
- **The 3-Step Contact Center Capacity Sizer Formula**:
  1. **Step 1: Convert Concurrent Sessions to Peak QPS**: Given an average human reading + typing cycle time of `40 seconds/turn`, `10,000` concurrent active sessions generate:
     $$\text{Peak QPS} = \frac{10,000 \text{ sessions}}{40 \text{ sec/turn}} = \mathbf{250 \text{ QPS}}$$
  2. **Step 2: Apply the `35 / 45 / 20` Offload Funnel**:
     - **Tier-0 (`35% = 87.5 QPS`)**: Top repetitive FAQs hit **Parameterized `Redis Semantic Cache`**, requiring **`0 GPUs`** (`8ms–15ms` latency);
     - **Tier-1 (`45% = 112.5 QPS`)**: Order status, billing lookups, and single-policy questions route to **`Qwen2.5-8B / Llama-3.1-8B FP8`** (`~32 QPS` per `H200/H100` with prefix caching) $\rightarrow$ **`4x H200 GPUs`** (or `8x L40S`);
     - **Tier-2 (`20% = 50 QPS`)**: Multi-policy disputes and human-agent Whisper drafts route to **`70B FP8 / GPT-4o PTU`** (`~42 QPS` per `8x H200` node + N+2 HA) $\rightarrow$ **`10x H200 GPUs`** (or `90 PTUs`).
  3. **Step 3: Total Fleet**: **`14 GPU-equivalents`** (plus `2x L4` for parallel `TEI` reranking and guardrails) = **`$96,000/month`** (`$0.0016/turn`), saving **`$1,008,000/month`** compared to Tier 1's 64-GPU monolithic fleet!

### Q2: How do we prevent the `35%` Semantic Cache from leaking Customer A's tracking number or account balance to Customer B?
- **Parameterized Policy Templates + `< 15ms` Live Slot Hydration**:
  - Never store raw personalized tool outputs inside a shared vector cache!
  - Cache only **Static Policy Templates** keyed by `[tenant_id, intent_id, policy_version_hash]` containing placeholders (e.g., `{{live_order_status}}`). Upon a cache hit, the gateway hydrates the placeholder via a deterministic `< 15ms` REST/GraphQL call authenticated with the caller's `user_id`—guaranteeing **0% cross-user PII leakage** and instant cache invalidation whenever `policy_version_hash` updates.

### Q3: What happens when a customer becomes frustrated (`sentiment < -0.6`) or cannot be resolved by the self-service bot?
- **`250ms` Structured Warm Handoff into Human-Agent Whisper Copilot Mode**:
  - Instead of dumping a raw 15-turn transcript onto a human agent, the orchestrator generates a **Structured Handoff Card** (`3-bullet issue summary + verified order/entity IDs + recommended resolution action`) into the CRM (`Genesys / Salesforce / ServiceNow`) in `250ms`, and switches the LLM into **Whisper Assist Mode** to draft real-time replies for the human agent (`93.8%` FCR).

---

## ③ 🧭 Visual Architecture Diagrams (Decision Tree + `35/45/20` Funnel Topology + `390ms` Parallel Fan-Out Sequence)

### 3.1 High-Concurrency Customer Service & Agent-Assist Copilot Selection Decision Tree

```mermaid
flowchart TD
    A["10,000-Concurrency Customer Service Copilot Sizing<br/>(250 QPS Peak | 60M Turns/Mo | Target P99 TTFT <= 450ms)"] --> B{"How are Pre-LLM steps (Shield, Router, RAG) orchestrated?"}
    B -- "5-Step Serial Waterfall (Shield -> Intent -> RAG -> LLM -> Judge) (Tier 1/3)" --> C["⛔ 1,180ms–3,420ms P99 TTFT!<br/>790ms wasted in sequential pre/post hops before first token"]
    B -- "48ms Parallel Speculative Fan-Out (asyncio.gather) + Stream Gate (Tier 4)" --> D{"How is the 250 QPS workload distributed across models?"}
    D -- "100% on Frontier 70B / GPT-4o (Tier 1)" --> E["⛔ Requires 64 GPUs ($1.104M/Mo)!<br/>Burns frontier compute on repetitive FAQs & simple lookups"]
    D -- "100% on 8B SLM Only (Tier 2)" --> F["⛔ FCR Drops to 63.5%!<br/>Fails complex multi-policy disputes; spikes human agent rework"]
    D -- "35% Redis Template Cache + 45% 8B SLM + 20% 70B/PTU Funnel (Tier 4)" --> G["🟩 Tier 4: 48ms Parallel Fan-Out + 35/45/20 Funnel + Whisper Handoff 🏆<br/>(390ms P99 TTFT | 93.8% FCR | 14 GPUs | -91.3% Monthly TCO)"]
    style G fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#f8fafc
    style C fill:#7f1d1d,stroke:#ef4444,color:#fecaca
    style E fill:#7f1d1d,stroke:#ef4444,color:#fecaca
    style F fill:#7f1d1d,stroke:#ef4444,color:#fecaca
```

### 3.2 Tier 4 `48ms` Parallel Pre-LLM Fan-Out & `35/45/20` Capacity Funnel Reference Architecture

```mermaid
flowchart LR
    subgraph FanOut["1. 48ms Parallel Pre-LLM Fan-Out (asyncio.gather)"]
        Traffic["💬 10,000 Sessions<br/>(250 QPS Peak)"] --> S1["🛡️ Input Prompt Shield (18ms)"]
        Traffic --> S2["🧭 INT8 Intent & Sentiment Router (12ms)"]
        Traffic --> S3["⚡ Speculative Hybrid RAG + TEI Rerank (48ms)"]
    end

    subgraph Tiers["2. 35 / 45 / 20 Compute Offload Funnel (14 GPU-Equivalents Total)"]
        S2 -->|35% Repetitive FAQs (87.5 QPS)| T0["🟢 Tier-0: Redis Policy Template + Live Slot Hydration<br/>(0 GPUs | 15ms Latency | 0% PII Leak)"]
        S2 -->|45% Standard Order/Policy (112.5 QPS)| T1["🔵 Tier-1: Qwen2.5-8B / Llama-8B FP8 Pool<br/>(4x H200 / 8x L40S | 280ms P99 TTFT)"]
        S2 -->|20% Complex Disputes & Whisper (50 QPS)| T2["🟣 Tier-2: 70B FP8 / GPT-4o PTU Pool<br/>(10x H200 / 90 PTU | 390ms P99 TTFT)"]
    end

    subgraph Output["3. Streaming NLI Guard & CRM Whisper Handoff"]
        T0 & T1 & T2 --> Gate["🔍 20-Token Sliding-Window Stream Gate<br/>+ 250ms Structured CRM Warm Handoff"]
    end
```

### 3.3 End-to-End Sequence: `390ms` First-Token Response via Parallel Speculative Pre-LLM Fan-Out

```mermaid
sequenceDiagram
    autonumber
    participant User as Customer (Omnichannel Chat)
    participant Orch as Async Copilot Orchestrator
    participant Pre as Parallel Pre-LLM (Shield + Router + TEI RAG)
    participant LLM as Tiered LLM Pool (Cache / 8B / 70B)
    participant CRM as Human Agent Desktop (Whisper Copilot)

    User->>Orch: 1. Send message: "My cold-chain grocery order arrived thawed—how does VIP refund work?"
    par t = 0ms: Parallel Fan-Out (Total elapsed = max(18ms, 12ms, 48ms) = 48ms!)
        Orch->>Pre: Task A: Run Prompt Shield safety check (18ms)
        Orch->>Pre: Task B: Classify intent=Complex_VIP_Claim (Tier-2 70B) & sentiment=-0.4 (12ms)
        Orch->>Pre: Task C: Speculative Hybrid RAG + L4 TEI Rerank for VIP Cold-Chain Policy (48ms)
    end
    Pre-->>Orch: 2. t = 48ms: Return safety pass, Tier-2 route, and Top-3 reranked policy chunks
    Orch->>LLM: 3. Dispatch to Tier-2 (70B FP8 / PTU) with 20-token sliding-window stream NLI check
    LLM-->>User: 4. t = 390ms: Emit first streaming token (8.8x faster than Tier 1's 3,420ms waterfall)!
    opt If sentiment drops below -0.6 or customer requests human agent
        Orch->>CRM: 5. Push structured summary + verified order slots to CRM in 250ms & enter Whisper Mode!
    end
```

---

## ④ 📊 4-Tier Architecture Decision Matrix

| Evaluation Dimension | 🟥 Tier 1: Serial `70B` Waterfall | 🟨 Tier 2: 100% `8B SLM` Only | 🟦 Tier 3: Serial `8B/70B` Router | 🟩 Tier 4: `35/45/20` Funnel + `48ms` Parallel RAG |
|---|---|---|---|---|
| **End-to-End `P99 TTFT` Latency** | `3,420.0 ms` ✕ (severe freeze) | `480.0 ms` | `1,180.0 ms` ✕ (790ms serial hops) | 🏆 **`390.0 ms` (`48ms` parallel fan-out + stream gate)** |
| **First-Contact Resolution (`FCR`)** | `81.0%` | `63.5%` ✕ (fails complex claims) | `88.5%` | 🏆 **`93.8%` (tiered routing + Whisper warm handoff)** |
| **Cost per 1,000 Customer Turns** | `$18.40 / 1K` ✕ | `$1.85 / 1K` | `$5.20 / 1K` | 🏆 **`$1.60 / 1K` (`$0.0016` per turn)** |
| **Required `GPU/PTU` Fleet (`250 QPS`)** | `64 GPUs` ✕ | `12 GPUs` | `28 GPUs` | 🏆 **`14 GPUs` (`4x 8B + 10x 70B` — `-78.1%` fleet)** |
| **60M-Turn Monthly TCO** | `$1,104,000 / mo` ✕ | `$385,000 / mo` (incl. escalations) | `$312,000 / mo` | **`$96,000 / mo` 🏆 (-91.3% vs Tier 1)** |

---

## ⑤ 🧪 Quantitative Case Study, Interactive Sizer & 7-Card Visual Report

- **Interactive HTML Sizer**: 👉 [Open 10,000-Concurrency Customer Service & Agent-Assist Copilot Sizer](../reports/enterprise-multimodal-customer-service-copilot-architecture-sizing-sizer.html)
- **7-Card 4:5 Visual Benchmark Report**: 👉 [Open 7-Card Visual Benchmark Deck](../reports/enterprise-multimodal-customer-service-copilot-architecture-sizing-selection-cards.html)
- **Deterministic Verification**:
  ```bash
  python experiments/enterprise_multimodal_customer_service_copilot_architecture_sizing_selection/run.py --check
  python -m unittest tests/test_enterprise_multimodal_customer_service_copilot_architecture_sizing_selection.py
  ```

---

## ⑥ 🛠️ Production Readiness & Architecture Review Checklist

- [ ] **Collapse Input Guardrails, Intent Routing, and Speculative RAG into a Single `asyncio.gather` Parallel Fan-Out (`< 50ms`)**: Eliminate sequential pre-LLM waterfall delays.
- [ ] **Size GPU / PTU Capacity Using the `35% Template Cache / 45% 8B SLM / 20% 70B Frontier` Funnel**: Convert concurrent sessions via `Peak QPS = Concurrent_Sessions / 40s` and reserve frontier compute strictly for the 20% complex tail.
- [ ] **Separate Static Policy Templates from Dynamic User Slot Hydration in the Semantic Cache**: Key cache entries by `policy_version_hash` and hydrate personal order/account fields live (`< 15ms`) to guarantee zero cross-user PII leakage.
- [ ] **Replace Blocking Full-Response Output Judges with 20-Token Sliding-Window Streaming Moderation**: Maintain sub-400ms TTFT while catching policy violations mid-stream.
- [ ] **Implement `250ms` Structured Warm Handoff into CRM Whisper Assist Mode**: Pass summarized context, verified entity IDs, and suggested actions directly to human agents when sentiment or complexity triggers escalation.
