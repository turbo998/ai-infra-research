# Enterprise Multi-Tenant SaaS AI Tiered Rate-Limiting (`GCRA` Cell Rate), Asymmetric Input/Output Compute-Credit Metering (`0.15x / 1.0x / 4.0x`), Pre-Flight Stream Escrow & Work-Conserving Burst-Borrowing Architecture Selection Guide: Fixed-Window `HTTP RPM` vs. Flat Unweighted `TPM` Cap vs. Post-Stream Local Token Bucket vs. Distributed `Redis-Cell GCRA` Pre-Flight Escrow + Asymmetric Compute Credits + Opportunistic Burst-Borrowing Queue

> **Language / 语言**: **English (Current)** | [中文](./enterprise-saas-ai-tiered-rate-limiting-gcra-token-credit-burst-selection-faq.md)  
> **Scope**: End-to-end production reference architecture and TCO selection guide for Cloud & AI Solution Architects, Multi-Tenant B2B SaaS Platform Leads, Internal Enterprise MaaS Architects, and API Gateway Engineers.  
> **Provenance**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`). Deterministic evaluation model sealed in [`experiments/enterprise_saas_ai_tiered_rate_limiting_gcra_token_credit_burst_selection/results.json`](../../experiments/enterprise_saas_ai_tiered_rate_limiting_gcra_token_credit_burst_selection/results.json).

![Enterprise Multi-Tenant SaaS AI Tiered Rate-Limiting GCRA, Compute Credits & Burst Borrowing Summary](../../assets/svg/enterprise-saas-ai-tiered-rate-limiting-gcra-token-credit-burst-selection-faq-summary.svg)

---

## ① 📌 Executive Summary (3-Minute Architect Verdict)

Operating a multi-tenant B2B SaaS AI platform serving **5,000 enterprise customer organizations (`Free / Pro / Enterprise` tiers) across a shared `64-GPU NVIDIA H100` cluster (`85M tokens/min` peak throughput behind `12` stateless API gateway replicas)** exposes three critical failure modes in traditional API rate-limiting:
1. **HTTP Request-Per-Minute (`RPM`) Fixed-Window Counters (Tier 1) Suffer `:00` Minute-Boundary Stampedes and Equate a 200-Token Chat with a `64K-In + 8K-Out` Reasoning Loop (`4,120 ms` Noisy-Neighbor P99 TTFT)**.
2. **Flat Unweighted Total-Token (`TPM`) Hard Caps (Tier 2) Penalize Prefix-Cached RAG Tenants (`19.2%` False `HTTP 429` Throttling) While Leaving `49%` of the Shared GPU Cluster Idle**.
3. **Post-Stream Deduction Token Buckets (Tier 3) Allow 50 Concurrent SSE Streams to Bypass Quota Gates Before the First Stream Completes (`6.2%–16.4%` Quota Overrun Leakage)**.

Evaluated across **5,000 B2B SaaS tenant organizations, 64 H100 GPUs, 12 gateway replicas, and 85M tokens/min**, the four multi-tenant rate-limiting architectures compare as follows:

| Architectural Tier | Core Stack Combination | Noisy-Neighbor P99 TTFT | False `HTTP 429` on Cached RAG | Shared GPU Cluster Utilization | Quota Overrun Leakage | Monthly Shared GPU Fleet TCO | Production Verdict |
|---|---|---|---|---|---|---|---|
| **🟥 Tier 1: HTTP `RPM` Fixed-Window Counter** | 60-second fixed request counter (ignores token count & boundary bursts) | `4,120 ms` ✕ | `24.5%` ✕ | `42.0%` ✕ *(massive headroom waste)* | `16.4%` ✕ *(:00 2x stampede)* | `$165,000 / mo` ✕ | ⛔ 1 long reasoning loop starves 150 chats |
| **🟨 Tier 2: Flat Unweighted Total-`TPM` Hard Cap** | 1 cached token == 1 uncached token == 1 decode token; rigid ceilings | `2,680 ms` ✕ | `19.2%` ✕ *(penalizes cached RAG)* | `51.0%` ✕ *(49% GPUs sit idle)* | `9.8%` ✕ | `$138,000 / mo` ✕ | ⛔ Throttles cheap prefix hits & wastes GPUs |
| **🟦 Tier 3: Post-Stream Local Token Bucket** | Per-gateway memory bucket deducting tokens only after SSE `[DONE]` | `1,390 ms` | `8.4%` ✕ | `63.0%` | `6.2%` ✕ *(parallel SSE bypass)* | `$104,000 / mo` ✕ | ⚠️ 50 parallel streams bypass gate pre-finish |
| **🟩 Tier 4: Pre-Flight `GCRA` + Weighted Credits + Burst Borrow** | **Distributed `Redis-Cell GCRA` Pre-Flight Escrow + `0.15x/1x/4x` Compute Credits + `<75%` Work-Conserving Burst-Borrowing** | **`310 ms` 🏆** *(-92.5%)* | **`0.2%` 🏆** *(-24.3 pts)* | **`84.5%` 🏆** *(+42.5 pts)* | **`0.0%` 🏆** *(Zero leak)* | **`$62,000 / mo` 🏆** *(-62.4%)* | 🌟 **Recommended Multi-Tenant SaaS Standard** |

> 💡 **Golden Rule for Multi-Tenant SaaS & Enterprise MaaS Architects**: **"Never meter multi-tenant LLM traffic with flat 1:1 token counters or post-completion deductions—weight every token by its true GPU Roofline cost (`0.15x` cached prefix input, `1.0x` uncached prefill input, `4.0x` autoregressive decode output), atomically escrow `Prompt + P90 Expected Output` credits upfront via a continuous `GCRA` cell-rate limiter (settling exact differences on stream `[DONE]` or TCP abort), and allow bursting tenants to borrow idle GPU capacity opportunistically whenever cluster utilization is below `75%`."**

---

## ② 🎯 Customer & Architect Deep-Dive Q&A (3 Multi-Tenant SaaS Rate-Limiting Pain Points)

### Q1: Why is flat 1:1 `TPM` metering unfair to RAG workloads and blind to Decode bottlenecks, and how do `0.15x / 1.0x / 4.0x` Compute Credits fix it?
- **First-Principles Reality**:
  - On an H100 GPU, reading `30,000` warm prefix-cached tokens from HBM takes `<15 ms`, whereas generating `8,000` autoregressive decode tokens requires 8,000 sequential HBM weight reads over `25+ seconds`.
  - Flat `TPM` (Tier 2) charges a `32K` cached RAG request `32,200 tokens` (triggering `19.2%` false `HTTP 429` rejections) while charging an `8K` reasoning decode loop only `9,000 tokens`.
  - Tier 4 meters **Asymmetric Roofline Compute Credits**:
    $$C(\text{req}) = 0.15 \cdot T_{\text{cached\_in}} + 1.00 \cdot T_{\text{uncached\_in}} + 4.00 \cdot T_{\text{decode\_out}}$$
  - Under this weighting, the `32K` cached RAG request costs only **`6,500 credits`** (`0.2%` false 429 rate) while the heavy `8K` decode loop pays **`33,000 credits`**.

### Q2: How does Two-Phase Pre-Flight Escrow + Post-Stream `GCRA` Settlement eliminate the `6.2%` parallel SSE overrun loophole?
- **Upfront Credit Authorization + Stream-Close Refund**:
  1. **Pre-Flight Escrow on Request Arrival**: Before forwarding an SSE request to the GPU pool, the gateway tokenizes the prompt and adds the model's historical **`P90` output length** (capped by `max_tokens`), atomically advancing the tenant's **Generic Cell Rate Algorithm (`GCRA`) Theoretical Arrival Time (`TAT`)** in Redis-Cell (`0.0%` parallel bypass).
  2. **Post-Stream / TCP-Abort Settlement**: When the SSE stream emits `[DONE]`—or if the client disconnects mid-stream (`TCP RST/FIN`)—Envoy's stream-close hook refunds unused output credits $4.0 \times (T_{\text{est\_out}} - T_{\text{actual\_out}})$ back to the tenant's `TAT` ledger.

### Q3: How does Work-Conserving Burst-Borrowing boost shared GPU cluster utilization from `42.0%` to `84.5%` without hurting guaranteed SLAs?
- **Dual-Priority `Guaranteed` vs. `Opportunistic_Borrow` Admission**:
  - When a tenant exceeds its committed baseline rate limit while aggregate GPU cluster utilization is **`< 75%`**, Tier 4 does not drop the request with `HTTP 429`. Instead, it admits the request tagged `qos=Opportunistic_Borrow` into a lower-priority queue that yields immediately if guaranteed baseline traffic arrives—unlocking **`84.5%` cluster utilization** and cutting monthly GPU fleet TCO by **`62.4%`**.

---

## ③ 🧭 Visual Architecture Diagrams (Decision Tree + Pre-Flight GCRA Topology + Sequence Flow)

### 3.1 Architecture Selection Decision Tree

```mermaid
flowchart TD
    A["Enterprise Multi-Tenant B2B SaaS / Internal MaaS Rate-Limiting<br/>(5,000 Tenants + 64xH100 Shared Pool + 12 Gateways + 85M Tok/Min)"] --> B{"Uses HTTP RPM fixed-window counters<br/>or flat unweighted 1:1 Total-TPM caps?"}
    B -- "Yes (RPM or flat 1:1 TPM)" --> C["⛔ Noisy-Neighbor Stalls & Cached-RAG Throttling (Tier 1 / Tier 2)!<br/>4,120ms P99 TTFT, 19.2%–24.5% false 429s on cached RAG & 49% idle GPUs"]
    B -- "Uses token-bucket rate limiting" --> D{"Combines pre-flight GCRA escrow, asymmetric 0.15x/1x/4x<br/>compute credits & work-conserving burst borrowing?"}
    D -- "No (Post-stream deduction only)" --> E["⚠️ Parallel SSE Stream Overrun Loophole (Tier 3)<br/>50 concurrent requests bypass gate before first stream finishes (6.2% leak)"]
    D -- "Yes (Pre-Flight GCRA + Asymmetric Credits + Borrow)" --> F["🟩 Tier 4: Pre-Flight GCRA + 0.15x/1x/4x Credits + Burst Borrow 🏆<br/>(310 ms P99 TTFT | 0.2% False 429 | 84.5% GPU Util | -62.4% Fleet TCO)"]
    style F fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#f8fafc
    style C fill:#7f1d1d,stroke:#ef4444,color:#fecaca
```

### 3.2 Tier 4 Pre-Flight `GCRA` + Asymmetric Compute Credits + Burst-Borrowing Topology

```mermaid
flowchart LR
    subgraph Meter["1. Roofline Compute-Credit Calculator"]
        Req["Incoming Tenant Request<br/>(Free / Pro / Enterprise)"] --> Calc["⚖️ Asymmetric Credit Estimator<br/>0.15x Cached + 1.0x Uncached In<br/>+ 4.0x P90 Expected Decode Out"]
    end

    subgraph GCRA["2. Distributed GCRA Escrow & Borrow Gate"]
        Calc --> Gate["⏱️ Redis-Cell GCRA (Continuous TAT)<br/>Eliminates :00 Boundary Bursts<br/>Atomically Escrows Pre-Flight Credits"]
        Gate -->|Within Quota| Guar["🟢 Guaranteed Baseline Queue"]
        Gate -->|Over Quota & Cluster <75%| Borrow["🟡 Opportunistic Burst-Borrow Queue<br/>(Drives 84.5% GPU Utilization)"]
    end

    subgraph Settle["3. Post-Stream [DONE] / Abort Settlement"]
        Guar & Borrow --> GPU["🚀 Shared 64xH100 Pool"]
        GPU -->|SSE [DONE] or TCP RST Hook| Refund["💳 Settles Actual Generated Tokens<br/>Refunds Unused Escrow (0.0% Leak)"]
        Refund -.-> Gate
    end
```

### 3.3 End-to-End Sequence: Pre-Flight Escrowing a `32K` Cached RAG Request & Settling on Stream `[DONE]`

```mermaid
sequenceDiagram
    autonumber
    participant Tenant as Enterprise SaaS Client (Pro Tier)
    participant Envoy as Envoy AI Gateway (12 Replicas)
    participant GCRA as Distributed GCRA Ledger (Redis-Cell)
    participant Pool as Shared 64xH100 Pool (58% Busy)

    Tenant->>Envoy: POST /v1/chat/completions stream=true (30K cached prefix + 2K new input, P90 expected out = 400 tokens)
    Envoy->>Envoy: Compute pre-flight credits: 0.15*(30,000) + 1.0*(2,000) + 4.0*(400) = 8,100 Credits (vs. 32,400 raw tokens!)
    Envoy->>GCRA: Execute atomic GCRA pre-flight escrow (8,100 Credits) & advance continuous TAT timestamp
    GCRA-->>Envoy: If baseline full but Pool < 75% busy -> issue qos="Opportunistic_Borrow" admission token!
    Envoy->>Pool: Forward streaming request to GPU pool (P99 TTFT = 310 ms)
    Pool-->>Envoy: Stream completes `[DONE]` with 195 actual decode tokens generated (vs. 400 estimated)
    Envoy->>GCRA: Trigger stream-close settlement hook -> refund 4.0 * (400 - 195) = +820 Credits to tenant TAT!
```

---

## ④ 📊 4-Tier Architecture Decision Matrix

| Evaluation Dimension | 🟥 Tier 1: Fixed-Window `RPM` | 🟨 Tier 2: Flat Unweighted `TPM` | 🟦 Tier 3: Post-Stream Local Bucket | 🟩 Tier 4: Pre-Flight `GCRA` + Weighted Credits |
|---|---|---|---|---|
| **Noisy-Neighbor P99 TTFT Latency** | `4,120 ms` ✕ (`:00` burst + 64K hogs) | `2,680 ms` ✕ (decode saturation) | `1,390 ms` | 🏆 **`310 ms` (`-92.5%` vs. baseline)** |
| **False `HTTP 429` on Cached RAG** | `24.5%` ✕ | `19.2%` ✕ (cached charged at 1.0x) | `8.4%` ✕ | 🏆 **`0.2%` (`0.15x` cached prefix credit discount)** |
| **Concurrent Stream Overrun Leakage** | `16.4%` ✕ (`:00` 2x boundary burst) | `9.8%` ✕ | `6.2%` ✕ (parallel pre-finish bypass) | 🏆 **`0.0%` (atomic pre-flight escrow + abort hook)** |
| **Shared GPU Cluster Utilization** | `42.0%` ✕ (idle headroom buffer) | `51.0%` ✕ (rigid non-borrowing cap) | `63.0%` | 🏆 **`84.5%` (`<75%` work-conserving burst borrowing)** |
| **Monthly Shared GPU Fleet TCO** | `$165,000 / mo` | `$138,000 / mo` | `$104,000 / mo` | **`$62,000 / mo` 🏆 (-62.4%)** |

---

## ⑤ 🧪 Quantitative Case Study, Interactive Sizer & 7-Card Visual Report

- **Interactive HTML Sizer**: 👉 [Open Enterprise Multi-Tenant SaaS AI Tiered Rate-Limiting (GCRA) & Credit Sizer](../reports/enterprise-saas-ai-tiered-rate-limiting-gcra-token-credit-burst-sizer.html)
- **7-Card 4:5 Visual Benchmark Report**: 👉 [Open 7-Card Visual Benchmark Deck](../reports/enterprise-saas-ai-tiered-rate-limiting-gcra-token-credit-burst-selection-cards.html)
- **Deterministic Verification**:
  ```bash
  python experiments/enterprise_saas_ai_tiered_rate_limiting_gcra_token_credit_burst_selection/run.py --check
  python -m unittest tests/test_enterprise_saas_ai_tiered_rate_limiting_gcra_token_credit_burst_selection.py
  ```

---

## ⑥ 🛠️ Production Readiness & Architecture Review Checklist

- [ ] **Replace Request-Per-Minute (`RPM`) and Unweighted 1:1 `TPM` Counters with Asymmetric Compute Credits (`0.15x` Cached / `1.0x` Uncached In / `4.0x` Decode Out)**: Align tenant rate limits with actual GPU Roofline compute and HBM bandwidth consumption.
- [ ] **Enforce Two-Phase Pre-Flight Escrow + Post-Stream Settlement on All Streaming (`SSE`) Endpoints**: Reserve `Prompt + P90 Expected Output` credits before dispatching to GPUs and refund unused credits upon `[DONE]`.
- [ ] **Attach Envoy Stream-Close Hooks to Settle Partial Generations on Client Disconnects (`TCP RST/FIN`)**: Prevent unbilled GPU exhaustion from aborted long-output requests.
- [ ] **Use Continuous `GCRA` (Theoretical Arrival Time `TAT`) Instead of 60-Second Fixed Windows**: Eliminate `:00` minute-boundary double-quota retry stampedes.
- [ ] **Enable Work-Conserving Opportunistic Burst-Borrowing Below `75%` Cluster Utilization**: Allow bursting tenants to utilize idle GPUs with automatic preemption when guaranteed baseline traffic arrives.
