# Enterprise LLM Customization Path (`Prompt + RAG` vs. `SFT / Multi-LoRA` vs. `Teacher-Student Distillation`) & Cloud GPU SKU Sizing Guide (`H200` vs. `H100` vs. `A100` vs. `L40S`)

> **Language / 语言**: **English (Current)** | [中文](./enterprise-llm-customization-rag-sft-lora-distillation-gpu-sizer-selection-faq.md)  
> **Scope**: End-to-end production reference architecture, customization decision tree, and GPU capacity sizer for Cloud & AI Solution Architects, Enterprise AI Platform Leads, and FinOps Capacity Planners.  
> **Provenance**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`). Deterministic evaluation model sealed in [`experiments/enterprise_llm_customization_rag_sft_lora_distillation_gpu_sizer_selection/results.json`](../../experiments/enterprise_llm_customization_rag_sft_lora_distillation_gpu_sizer_selection/results.json).

![Enterprise LLM Customization Path & Cloud GPU SKU Sizing Summary](../../assets/svg/enterprise-llm-customization-rag-sft-lora-distillation-gpu-sizer-selection-faq-summary.svg)

---

## ① 📌 Executive Summary (3-Minute Architect Verdict)

When scaling an enterprise AI platform across **20 business units (Legal, Tax, HR, Support, Engineering) at `250 QPS` aggregate throughput (`80%` high-volume structured extraction/classification/SQL, `20%` complex multi-hop reasoning)**, Solution Architects must answer two foundational questions: **(1) When do we use `Prompt + RAG` vs. `LoRA Fine-Tuning` vs. `Model Distillation`?** and **(2) Which cloud GPU SKU (`H200`, `H100`, `A100`, or `L40S`) fits each workload?** Three common mistakes burn millions in GPU spend:
1. **Mistake #1: Full-Weight `70B SFT` to Memorize Dynamic Enterprise Facts + Isolated Departmental GPU Clusters (Tier 1, `62.4%` Accuracy, `$245,000/month`)**: Fine-tuning weights on daily-changing product manuals hallucinates stale facts, breaks citation grounding, violates document-level ACLs, and wastes `64x H100` GPUs across 20 siloed deployments;
2. **Mistake #2: Stuffing `6,000 Tokens` of Few-Shot Examples into Every `70B` Prompt Instead of Fine-Tuning/Distilling (Tier 2, `1,820ms` P99 Latency, `$142,000/month`)**: Paying to prefill 6,000 tokens of schema/style examples on every single call requires `32x H100` GPUs just to keep up with prefill compute;
3. **Mistake #3: GPU SKU Mismatch**: Running `8B` models on underutilized `H100`s, or forcing `70B` models across `4x L40S` PCIe cards with severe tensor-parallel communication overhead.

Evaluated across **20 business units and 250 QPS aggregate production load**, the four customization and GPU sizing tiers compare as follows:

| Architectural Tier | Core Customization & GPU Stack | Domain & Fresh-Fact Accuracy | Schema & Syntax Adherence | End-to-End P99 Latency | GPU Fleet Footprint (`250 QPS`) | Monthly Train & Serve TCO | Production Verdict |
|---|---|---|---|---|---|---|---|
| **🟥 Tier 1: Isolated `70B SFT` Clusters (No RAG)** | Full-weight SFT on dynamic docs + 20 separate 70B clusters | `62.4%` ✕ *(stale facts, no ACLs)* | `94.0%` | `1,480 ms` | `64x H100 (80GB)` ✕ | `$245,000 / mo` ✕ | ⛔ #1 customization anti-pattern |
| **🟨 Tier 2: `6K Few-Shot Prompt + RAG` on `70B`** | Stuffs 6,000 few-shot tokens + 4,000 RAG tokens per call | `89.5%` | `84.2%` ✕ *(schema drift)* | `1,820 ms` ✕ *(prefill bloat)* | `32x H100 (80GB)` | `$142,000 / mo` ✕ | ⚠️ Fine for Day-1 PoC; too slow/costly at scale |
| **🟦 Tier 3: `RAG` + Shared `70B Multi-LoRA` (`H200`)** | RAG for facts + 20 rank-32 LoRA adapters sharing one 70B base | `95.8%` | `98.5%` | `640 ms` | `8x H200 (141GB)` | `$54,000 / mo` | ✔️ Eliminates 20x cluster sprawl & few-shot bloat |
| **🟩 Tier 4: `RAG` + Distilled `8B` (`L40S`) + `70B Multi-LoRA` (`H200`)** | **RAG for facts + `80%` routine tasks on distilled `8B (8x L40S)` + `20%` complex tasks on `70B Multi-LoRA (4x H200)`** | **`96.6%` 🏆** | **`99.1%` 🏆** *(internalized schema)* | **`245 ms` 🏆** *(-86.5%)* | **`8x L40S + 4x H200` 🏆** | **`$28,500 / mo` 🏆** *(-88.4%)* | 🌟 **Recommended Enterprise Customization Standard** |

> 💡 **Golden Rule for Enterprise LLM Customization & GPU Sizing**: **"Follow the Orthogonal Customization Law: `RAG` is for dynamic knowledge, citations, and per-user ACLs (`What to know`); `LoRA SFT and Teacher-Student Distillation` are for output schema, domain syntax, and compressing 6,000-token few-shot examples into weights (`How to act`). Distill 80% of high-volume structured tasks from a 70B Teacher into an `8B FP8` Student running on `NVIDIA L40S (48GB)`, and serve the remaining 20% complex reasoning tasks from a single shared `70B FP8 Multi-LoRA` pool on `NVIDIA H200 (141GB)`."**

---

## ② 🎯 Customer & Architect Deep-Dive Q&A (3 Customization & GPU Sizing Pain Points)

### Q1: How do `RAG`, `Multi-LoRA SFT`, and `Teacher-Student Distillation` work together rather than competing?
- **Three-Stage Enterprise Maturity Flywheel**:
  1. **RAG for Fresh Facts & Access Control**: Dynamic documents, prices, and policies stay in the hybrid vector index with row-level ACL filters.
  2. **Multi-LoRA SFT to Eliminate Few-Shot Prompt Bloat & Cluster Sprawl**: Training a rank-32 LoRA adapter (`~160 MB`) per department internalizes 6,000 tokens of few-shot format/style examples into weights (cutting input prompt tokens by **93%**). Using `vLLM / SGLang` dynamic `Multi-LoRA` (`S-LoRA`), all 20 departmental adapters share a single `70B` base model in GPU memory.
  3. **70B-to-8B Distillation for High-QPS Routine Workloads**: Harvesting 20,000 verified gold responses from the 70B teacher to fine-tune an `8B FP8` student (`Llama-3.1-8B / Qwen2.5-7B`) achieves `99.1%` schema adherence at `245ms` P99 latency and 1/8th the compute cost.

### Q2: How should architects map `8B`, `32B`, and `70B+` models across `H200 (141GB)`, `H100 (80GB)`, `A100 (80GB)`, and `L40S (48GB)` GPUs?
- **First-Principles VRAM Capacity & Bandwidth Mapping**:
  - **`8B–14B FP8` Models (`~8.5–15 GB` weights) -> `1x NVIDIA L40S (48GB GDDR6, 864 GB/s)`**: Leaves `32–36 GB` of VRAM for KV-cache and supports native Ada FP8 Tensor Cores at the lowest cost per token.
  - **`70B FP8` Models (`~72 GB` weights) -> `1x NVIDIA H200 (141GB HBM3e, 4.8 TB/s)` (`TP=1`) or `2x H100 80GB` (`TP=2`)**: On an `80 GB H100`, 72 GB of weights leave `< 6 GB` for KV-cache, forcing `TP=2` across two GPUs. A single **`141 GB H200`** holds the entire 72 GB model **plus `60+ GB` of KV-cache and 20 LoRA adapters on one GPU (`TP=1`)**, eliminating cross-GPU AllReduce overhead and boosting per-GPU throughput by **`+28.0%`**!

---

## ③ 🧭 Visual Architecture Diagrams (Decision Tree + Heterogeneous GPU Topology + Sequence Flow)

### 3.1 Enterprise LLM Customization & GPU SKU Selection Decision Tree

```mermaid
flowchart TD
    A["Enterprise LLM Customization & GPU Capacity Sizing<br/>(20 Business Units + 250 QPS Aggregate Load)"] --> B{"Goal is injecting dynamic facts / citations / ACLs,<br/>or enforcing strict JSON/SQL schema & compressing 6K few-shot prompts?"}
    B -- "Dynamic facts / Citations / Per-user ACLs" --> C["📚 Always Use RAG (Hybrid Search + Rerank)<br/>Never bake daily-changing facts or ACL docs into model weights!"]
    B -- "Strict schema / Domain syntax / Eliminating 6K few-shot bloat" --> D{"Task QPS volume and reasoning complexity?"}
    D -- "Complex multi-hop reasoning across 20 departments (20% load)" --> E["🧩 Shared 70B FP8 + Dynamic Multi-LoRA Pool<br/>Run TP=1 on NVIDIA H200 (141GB)<br/>(20 LoRA adapters share 1 base model in memory)"]
    D -- "High-QPS structured extraction / classification / SQL (80% load)" --> F["⚡ 70B Teacher -> 8B FP8 Student Distillation 🏆<br/>Run on NVIDIA L40S (48GB) + RAG Context<br/>(245ms P99 Latency | 99.1% Schema | -88.4% Monthly TCO)"]
    style E fill:#1e3a8a,stroke:#3b82f6,stroke-width:2px,color:#f8fafc
    style F fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#f8fafc
```

### 3.2 Tier 4 `RAG` + `L40S 8B Distilled Student Pool` + `H200 70B Multi-LoRA Pool` Reference Architecture

```mermaid
flowchart LR
    subgraph Gateway["1. AI Gateway & ACL-Aware RAG"]
        Req["20 Business Units (250 QPS)"] --> RAG["🔍 ACL-Aware Hybrid RAG<br/>Injects fresh facts & citations"]
        RAG --> Router["🧭 Task Complexity Router<br/>80% Routine vs 20% Complex"]
    end

    subgraph Serving["2. Right-Sized Heterogeneous GPU Pools"]
        Router -->|80% Routine JSON/SQL (200 QPS)| L40S["⚡ 8x NVIDIA L40S (48GB) Pool<br/>Serves Distilled 8B FP8 Student<br/>(Zero 6K Few-Shot Bloat | 245ms P99)"]
        Router -->|20% Deep Reasoning (50 QPS)| H200["🚀 4x NVIDIA H200 (141GB) Pool<br/>Single-GPU TP=1 70B FP8 Base<br/>+ 20 Hot-Swapped Dept LoRAs"]
    end

    subgraph Flywheel["3. Teacher-Student Distillation Flywheel"]
        H200 -.->|Harvests Verified Gold Traces| Distill["🧪 Offline Distillation & LoRA Pipeline"]
        Distill -.->|Updates 8B Student & LoRA Adapters| L40S
    end
```

### 3.3 End-to-End Sequence: Routing Between Distilled `8B` (`L40S`) and Shared `70B Multi-LoRA` (`H200`)

```mermaid
sequenceDiagram
    autonumber
    participant User as Business Unit Caller (`dept=tax` / `dept=claims`)
    participant GW as AI Gateway + RAG Engine
    participant L40S as 8B FP8 Student Pool (NVIDIA L40S 48GB)
    participant H200 as 70B FP8 Multi-LoRA Pool (NVIDIA H200 141GB)

    User->>GW: Send request with department metadata
    GW->>GW: 1. Retrieve fresh policy/pricing chunks via ACL-filtered RAG
    alt 80% High-Volume Structured Task (Invoice JSON Extraction / Ticket Triage / SQL)
        GW->>L40S: Route to distilled 8B FP8 Student (6,000-token few-shot examples internalized -> 400-token prompt)
        L40S-->>User: Return 99.1% schema-valid JSON in 245ms P99!
    else 20% Complex Multi-Hop Reasoning Task
        GW->>H200: Route to H200 70B shared base with `model="llama-70b-lora-tax-v3"`
        H200-->>User: vLLM S-LoRA batches the 160MB tax LoRA adapter dynamically on a single 141GB H200!
    end
```

---

## ④ 📊 Cloud GPU SKU Mapping & 4-Tier Architecture Decision Matrix

### 4.1 Cloud Inference GPU SKU Selection Matrix (`H200` vs. `H100` vs. `A100` vs. `L40S`)

| Cloud GPU SKU | Memory (`VRAM`) | Bandwidth (`TB/s`) | Native `FP8` | Sweet-Spot Models & Workloads | Key Sizing Insight |
|---|---|---|---|---|---|
| **`NVIDIA H200 SXM`** | **`141 GB HBM3e`** | **`4.8 TB/s`** | ✅ Yes | 🏆 **`70B–72B FP8` on a single GPU (`TP=1`)**, long-context (`32K–128K`), shared `Multi-LoRA` pools, or 8-GPU `671B MoE FP8` | Holds 72GB weights + 60GB+ KV-cache on 1 chip (`+28%` tok/s per GPU vs TP=2 H100) |
| **`NVIDIA H100 SXM`** | `80 GB HBM3` | `3.35 TB/s` | ✅ Yes | **`32B FP8` single-GPU**, or **`2x H100 (TP=2)` for `70B FP8`** high-throughput serving | 70B FP8 needs `TP=2` on 80GB cards so KV-cache doesn't starve |
| **`NVIDIA A100 SXM`** | `80 GB HBM2e` | `2.0 TB/s` | ❌ No | Legacy `BF16` or `INT4/AWQ` offline batch processing when Hopper is unavailable | Lacks native FP8; 2.4x lower bandwidth than H200 |
| **`NVIDIA L40S PCIe`** | **`48 GB GDDR6`** | `0.86 TB/s` | ✅ Yes | 🏆 **`8B–14B FP8` distilled students**, `Docling/MinerU` PDF layout, `Llama-Guard` & rerankers | Best $/token for `<= 14B` models; avoid 4-card PCIe TP on 70B |

### 4.2 4-Tier Customization & Serving Comparison (20 Business Units / 250 QPS)

| Evaluation Dimension | 🟥 Tier 1: 20 Isolated `70B SFT` Clusters | 🟨 Tier 2: `6K Few-Shot + RAG` (`70B`) | 🟦 Tier 3: `RAG + 70B Multi-LoRA` | 🟩 Tier 4: `RAG` + `8B Distill (L40S)` + `70B LoRA (H200)` |
|---|---|---|---|---|
| **Domain & Fresh-Fact Accuracy** | `62.4%` ✕ (stale weights) | `89.5%` | `95.8%` | 🏆 **`96.6%` (RAG for fresh facts + SFT/distill for task skill)** |
| **Schema & Syntax Adherence** | `94.0%` | `84.2%` ✕ | `98.5%` | 🏆 **`99.1%` (internalized schema weights)** |
| **End-to-End P99 Latency** | `1,480 ms` | `1,820 ms` ✕ (6K prefill bloat) | `640 ms` | 🏆 **`245 ms` (93% shorter prompt + 8B student speed)** |
| **GPU Fleet Size (`250 QPS`)** | `64x H100` ✕ | `32x H100` | `8x H200` | 🏆 **`8x L40S (48GB) + 4x H200 (141GB)`** |
| **Monthly Train & Serve TCO** | `$245,000 / mo` | `$142,000 / mo` | `$54,000 / mo` | **`$28,500 / mo` 🏆 (-88.4%, saves $2.59M/yr)** |

---

## ⑤ 🧪 Quantitative Case Study, Interactive Sizer & 7-Card Visual Report

- **Interactive HTML Sizer**: 👉 [Open Enterprise LLM Customization Path & Cloud GPU SKU Sizer](../reports/enterprise-llm-customization-rag-sft-lora-distillation-gpu-sizer.html)
- **7-Card 4:5 Visual Benchmark Report**: 👉 [Open 7-Card Visual Benchmark Deck](../reports/enterprise-llm-customization-rag-sft-lora-distillation-gpu-sizer-selection-cards.html)
- **Deterministic Verification**:
  ```bash
  python experiments/enterprise_llm_customization_rag_sft_lora_distillation_gpu_sizer_selection/run.py --check
  python -m unittest tests/test_enterprise_llm_customization_rag_sft_lora_distillation_gpu_sizer_selection.py
  ```

---

## ⑥ 🛠️ Production Readiness & Architecture Review Checklist

- [ ] **Enforce the Orthogonal Customization Split (`RAG` = Dynamic Facts & ACLs; `LoRA / Distillation` = Schema, Style & Prompt Compression)**: Never fine-tune model weights to memorize changing enterprise documents.
- [ ] **Consolidate Departmental Fine-Tuned Models onto a Shared `vLLM / SGLang` Dynamic `Multi-LoRA` Pool**: Serve dozens of rank-32 LoRA adapters from one shared base model instead of provisioning isolated GPU clusters per team.
- [ ] **Distill High-Volume Routine Tasks (`> 30 QPS` or `> 1,500` Few-Shot Tokens) from a `70B/Frontier` Teacher into an `8B FP8` Student on `L40S (48GB)` GPUs**.
- [ ] **Match GPU SKUs to Model Footprint (`8B–14B FP8 -> 1x L40S 48GB`; `70B FP8 -> 1x H200 141GB TP=1` or `2x H100 80GB TP=2`)**.
