# Enterprise High-Throughput Offline Document & Data Batch LLM Processing Architecture Selection Guide: Synchronous API Loops (`asyncio.gather` Anti-Pattern) vs. Cloud 50%-Off `Batch API` (`Azure OpenAI / Bedrock Batch`) vs. Unsorted `Ray Data + vLLM` vs. Off-Peak Idle `PTU` Backfill + Prefix-Sorted Spot `Ray/vLLM` + Cloud `Batch API`

> **Language / 语言**: **English (Current)** | [中文](./enterprise-offline-batch-llm-processing-architecture-selection-faq.md)  
> **Scope**: End-to-end production reference architecture and FinOps throughput guide for Cloud & AI Solution Architects, Data Engineering Platform Leads, and Enterprise AI Infrastructure Teams.  
> **Provenance**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`). Deterministic evaluation model sealed in [`experiments/enterprise_offline_batch_llm_processing_architecture_selection/results.json`](../../experiments/enterprise_offline_batch_llm_processing_architecture_selection/results.json).

![Enterprise Massive Offline Document & Data Batch LLM Architecture Summary](../../assets/svg/enterprise-offline-batch-llm-processing-architecture-selection-faq-summary.svg)

---

## ① 📌 Executive Summary (3-Minute Architect Verdict)

Alongside real-time interactive Copilots, enterprises run massive **offline batch LLM pipelines (`Nightly Batch ETL`)**—such as processing **`10,000,000` records every night (`60 Billion tokens/month`) within an 8-hour overnight window** for call-center transcript auditing, catalog classification, contract entity extraction, or compliance tagging. Three common anti-patterns waste millions and break online SLAs:
1. **Anti-Pattern #1: Hammering Live Synchronous APIs with `asyncio.gather()` Loops (Tier 1, `$3.80 / 1M tokens`, `38.5%` Online Copilot SLA Degradation, `$228,000/month`)**: Running a high-concurrency batch script against the same endpoint used by live users triggers `HTTP 429` rate-limit storms while paying full real-time synchronous token prices for non-urgent background work;
2. **Blind Spot #2: Leaving Already-Paid Nighttime `PTU` / Reserved GPU Valleys `85%` Idle While Paying Extra for Batch Compute (Tier 2 & Tier 3)**: Enterprises paying 24x7 for `Azure OpenAI PTUs` or reserved GPUs to cover daytime peaks often leave that capacity sitting empty from 20:00 to 07:00;
3. **Blind Spot #3: Feeding Offline `vLLM` Workers in Random Row Order (Tier 3, `< 8%` KV-Cache Hit Rate, `5.8h` Runtime)**: When 10M rows reference 5,000 shared system templates and catalog documents in random order, the GPU KV-cache thrashes on every row instead of reusing shared prefixes!

Evaluated across **60 Billion monthly batch tokens (`10,000,000` nightly records, 8-hour overnight SLA)**, the four batch architectures compare as follows:

| Architectural Tier | Core Offline Batch Stack | Effective Cost / 1M Tokens | Online Copilot `429` SLA Impact | 10M-Record Nightly Runtime | 24x7 Reserved Fleet Utilization | 60B-Token Monthly TCO | Production Verdict |
|---|---|---|---|---|---|---|---|
| **🟥 Tier 1: Sync Online API (`asyncio.gather`)** | Hammers live PAYGO/PTU endpoints in parallel loops | `$3.80 / 1M` ✕ | `38.5%` ✕ *(starves online TPM quota)* | `14.5 hrs` ✕ *(misses 8h SLA)* | `42.0%` | `$228,000 / mo` ✕ | ⛔ #1 batch anti-pattern; expensive & disruptive |
| **🟨 Tier 2: 100% Cloud 50%-Off `Batch API`** | Uploads JSONL to `Azure OpenAI / Bedrock Batch API` | `$1.90 / 1M` | `0.0%` *(separate quota pool)* | `9.2 hrs` | `65.0%` *(ignores night PTUs)* | `$114,000 / mo` | ✔️ Zero-ops & quota-safe; ideal for frontier models |
| **🟦 Tier 3: Unsorted On-Demand `Ray Data + vLLM`** | Dedicated on-demand GPUs processing rows in random order | `$1.25 / 1M` | `0.0%` | `5.8 hrs` | `74.0%` | `$75,000 / mo` | ⚠️ Misses `3.1x` prefix-sorting speedup & free night PTUs |
| **🟩 Tier 4: Idle Night `PTU` Backfill + Prefix-Sorted Spot `Ray/vLLM` + Cloud `Batch API`** | **Scavenges idle night `PTUs` (`$0` marginal) + Lexicographically prefix-sorted `Ray Data + vLLM` on Spot GPUs (`3.1x` faster) + Cloud `Batch API` overflow** | **`$0.58 / 1M` 🏆** *(-84.7%)* | **`0.0%` 🏆** *(`<100ms` P0 QoS yield)* | **`3.4 hrs` 🏆** *(4.3x faster)* | **`94.8%` 🏆** | **`$34,800 / mo` 🏆** *(-84.7%)* | 🌟 **Recommended Enterprise High-Throughput Batch Standard** |

> 💡 **Golden Rule for Enterprise Batch LLM Architects**: **"Combine three high-leverage mechanisms for offline batch pipelines: (1) never hammer synchronous online endpoints—use Cloud `50%-Off Batch API` (`Azure OpenAI Global Batch / AWS Bedrock Batch`) for frontier models; (2) scavenge already-paid nighttime `PTU` valleys (`20:00–07:00`) at `$0` marginal cost using a low-priority (`P2`) token-bucket controller that yields to online (`P0`) traffic in `< 100ms`; and (3) before running `Ray Data + vLLM` on Spot GPUs, sort the input table lexicographically by `(system_template_id, shared_doc_id)`—boosting GPU KV-cache hit rates from `8%` to `94%` and multiplying offline throughput by `3.1x`!"**

---

## ② 🎯 Customer & Architect Deep-Dive Q&A (3 Batch Contention, Idle PTU & GPU Throughput Pain Points)

### Q1: How do we harvest already-paid nighttime `PTU` capacity (`$0` marginal token cost) for batch jobs without ever degrading live user SLAs (`0.0%` impact)?
- **Priority-Aware Token-Bucket Scavenging (`P0` Online vs. `P2` Batch Backfill)**:
  - The AI Gateway continuously tracks real-time `P0` interactive PTU utilization $U_{\text{online}}$.
  - Whenever $U_{\text{online}} < 0.70$, the gateway's `P2` backfill worker drains nightly batch records from `Kafka / Azure Service Bus` at $\text{Rate}_{\text{batch}} = (0.85 - U_{\text{online}}) \times \text{TPM}_{\text{PTU}}$.
  - If an overseas office logs in and $U_{\text{online}}$ rises, the gateway throttles `P2` batch injection in **`< 100ms`**, giving 100% priority back to interactive users while processing **45% of the nightly batch for free**!

### Q2: Why does adding a single `ORDER BY (template_id, shared_doc_id)` step before `Ray Data + vLLM` speed up offline GPU batch processing by `3.1x`?
- **Exploiting Lexicographical Prefix Locality in GPU HBM**:
  - When 10,000,000 rows reference 5,000 shared product catalogs or legal contracts in random order, the GPU KV-cache evicts each 6,000-token document before the next row referencing it arrives (`< 8%` KV hit rate).
  - Sorting the dataset by `(template_id, shared_doc_id)` clusters all 2,000 rows sharing Contract A into the same continuous batch window—achieving a **`94.0%` GPU KV-cache hit rate** and cutting runtime from `5.8 hours` to **`3.4 hours`**.

### Q3: How do we protect multi-hour `Ray Data + vLLM` jobs on Spot GPUs from losing progress during a Spot preemption?
- **10,000-Row Micro-Batch Parquet Checkpointing**:
  - `Ray Data` streams input shards in 10,000-row micro-batches and commits completed output Parquet files atomically to `S3 / ADLS Gen2`. If a Spot GPU node is preempted at 80% progress, replacement workers resume from the last committed Parquet shard with `< 1%` recomputation.

---

## ③ 🧭 Visual Architecture Diagrams (Decision Tree + 3-Channel Hybrid Batch Topology + Sequence Flow)

### 3.1 Enterprise Offline Batch LLM Processing Selection Decision Tree

```mermaid
flowchart TD
    A["Enterprise Offline Batch LLM Processing Selection<br/>(60B Tokens/Month + 10M Nightly Records + 8h Overnight Window)"] --> B{"Currently calling synchronous online APIs in asyncio.gather() loops?"}
    B -- "Yes (Tier 1 Anti-Pattern)" --> C["⛔ Stop Immediately! Pays full price ($228K/mo)<br/>& triggers HTTP 429 storms that degrade online Copilots by 38.5%"]
    B -- "No (Dedicated Async Batch Architecture)" --> D{"Does the enterprise own daytime reserved PTUs or GPUs?"}
    D -- "Yes (85% idle overnight 20:00–07:00)" --> E["♻️ Priority 1: Scavenge Idle Nighttime PTUs (P2 QoS Backfill)<br/>Processes 40%–50% of nightly batch volume at $0 marginal cost!"]
    D -- "No idle PTUs or volume exceeds night PTU headroom" --> F{"Model family & completion SLA?"}
    E --> F
    F -- "Proprietary Frontier Model (GPT-4o / Claude) & 24h SLA" --> G["🟨 Cloud 50%-Off Asynchronous Batch API<br/>(Azure OpenAI Global Batch / AWS Bedrock Batch)"]
    F -- "Open-Weight / Distilled Model (8B–70B) & <4h Overnight SLA" --> H["🟩 Tier 4: Prefix-Sorted Ray Data + vLLM on Spot GPUs 🏆<br/>(94% KV Hit Rate | 3.1x Throughput | $0.58/1M Tokens | -84.7% TCO)"]
    style E fill:#1e3a8a,stroke:#3b82f6,stroke-width:2px,color:#f8fafc
    style H fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#f8fafc
    style C fill:#7f1d1d,stroke:#ef4444,color:#fecaca
```

### 3.2 Tier 4 Off-Peak `PTU` Backfill + Prefix-Sorted Spot `Ray/vLLM` + Cloud `Batch API` Architecture

```mermaid
flowchart LR
    subgraph Prep["1. Lexicographical Prefix Sorter & Sharder"]
        Raw["📦 10M Nightly Batch Rows<br/>(Parquet / Delta / Iceberg)"] --> Sort["🗂️ Prefix Locality Sorter<br/>ORDER BY (template_id, shared_doc_id)<br/>Clusters identical prefixes together"]
    end

    subgraph Channels["2. Three-Channel High-Throughput Execution"]
        Sort -->|Channel A: 45% Load ($0 Marginal Cost)| PTU["♻️ Nighttime Idle PTU Scavenger (P2 QoS)<br/>Injects only when online P0 load < 70%;<br/>Yields in <100ms on online traffic spikes"]
        Sort -->|Channel B: 45% Load (3.1x Faster)| Ray["⚡ Spot GPU (Ray Data + vLLM Offline)<br/>94% KV-Cache Hit Rate from prefix sort;<br/>Checkpoints Parquet every 10,000 rows"]
        Sort -->|Channel C: 10% Frontier Overflow| BatchAPI["☁️ Cloud 50%-Off Batch API<br/>(Azure OpenAI / Bedrock Batch)"]
    end
```

### 3.3 End-to-End Sequence: Completing 10 Million Nightly Records in `3.4 Hours` at `$0.58 / 1M Tokens`

```mermaid
sequenceDiagram
    autonumber
    participant Scheduler as Nightly Lakehouse Scheduler (20:00)
    participant Sorter as Ray Data Prefix Sorter
    participant Scav as Priority-Aware PTU Scavenger
    participant Spot as Spot GPU Ray Data + vLLM Pool
    participant Sink as Lakehouse Output Table (Parquet)

    Scheduler->>Sorter: Submit 10,000,000 nightly records
    Sorter->>Sorter: Sort by `(system_template_id, parent_doc_id)` & partition into 1,000 shards (10K rows/shard)
    par Dual-Engine Execution (Completes in 3.4 Hours)
        Sorter->>Scav: Dispatch 450 shards to Nighttime Idle PTU Scavenger
        Scav->>Sink: Backfill unused PTU headroom at $0 marginal cost with 0.0% online P0 SLA impact!
        Sorter->>Spot: Dispatch 550 prefix-sorted shards to Spot GPU Ray Data + vLLM pool
        Spot->>Sink: Achieve 94% GPU KV-cache hit rate (3.1x speedup) & commit atomic Parquet checkpoints every 10K rows!
    end
```

---

## ④ 📊 4-Tier Architecture Decision Matrix

| Evaluation Dimension | 🟥 Tier 1: Sync `asyncio.gather` | 🟨 Tier 2: 100% Cloud `Batch API` | 🟦 Tier 3: Unsorted On-Demand `Ray+vLLM` | 🟩 Tier 4: Idle `PTU` + Sorted Spot `Ray/vLLM` |
|---|---|---|---|---|
| **Effective Cost / 1M Tokens** | `$3.80 / 1M` ✕ (full sync rate) | `$1.90 / 1M` (50% discount) | `$1.25 / 1M` | 🏆 **`$0.58 / 1M` (45% free PTU backfill + Spot GPU)** |
| **Online Copilot SLA Impact** | `38.5%` ✕ (429 quota starvation) | `0.0%` (isolated quota) | `0.0%` | 🏆 **`0.0%` (`<100ms` P0 priority preemption)** |
| **10M-Record Nightly Runtime** | `14.5 hrs` ✕ | `9.2 hrs` | `5.8 hrs` (KV cache thrashing) | 🏆 **`3.4 hrs` (94% KV hit rate via prefix sorting)** |
| **24x7 Fleet Utilization** | `42.0%` ✕ | `65.0%` | `74.0%` | 🏆 **`94.8%` (harvests nighttime PTU valleys)** |
| **60B-Token Monthly TCO** | `$228,000 / mo` | `$114,000 / mo` | `$75,000 / mo` | **`$34,800 / mo` 🏆 (-84.7%, saves $2.31M/yr)** |

---

## ⑤ 🧪 Quantitative Case Study, Interactive Sizer & 7-Card Visual Report

- **Interactive HTML Sizer**: 👉 [Open Enterprise Offline Document & Data Batch LLM Processing Sizer](../reports/enterprise-offline-batch-llm-processing-architecture-sizer.html)
- **7-Card 4:5 Visual Benchmark Report**: 👉 [Open 7-Card Visual Benchmark Deck](../reports/enterprise-offline-batch-llm-processing-architecture-selection-cards.html)
- **Deterministic Verification**:
  ```bash
  python experiments/enterprise_offline_batch_llm_processing_architecture_selection/run.py --check
  python -m unittest tests/test_enterprise_offline_batch_llm_processing_architecture_selection.py
  ```

---

## ⑥ 🛠️ Production Readiness & Architecture Review Checklist

- [ ] **Ban High-Concurrency `asyncio.gather()` Batch Scripts Against Live Interactive Endpoints**: Isolate offline ETL workloads onto dedicated batch queues and asynchronous endpoints.
- [ ] **Deploy a Priority-Throttled (`P2`) Nighttime `PTU` Scavenger at the AI Gateway**: Harvest already-paid off-peak PTU capacity whenever live `P0` utilization is `< 70%`, yielding within `100ms` if interactive traffic rises.
- [ ] **Sort All Offline Batch Datasets Lexicographically by `(system_template_id, shared_context_doc_id)` Before GPU Execution**: Maximize continuous-batch Prefix KV Cache hits (`94%`) and multiply `Ray Data + vLLM` throughput by `3.1x`.
- [ ] **Commit Atomic Parquet Checkpoints Every `10,000 Rows` on Spot GPU Workers**: Guarantee zero lost work across Spot node preemptions.
- [ ] **Use Cloud `50%-Off Batch API` (`Azure OpenAI Global Batch / AWS Bedrock Batch`) for Proprietary Frontier Model Workloads that Tolerate a 24-Hour Window**.
