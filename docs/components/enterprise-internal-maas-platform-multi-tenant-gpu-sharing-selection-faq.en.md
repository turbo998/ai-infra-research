# Enterprise Multi-Team Shared Model-as-a-Service (`Internal MaaS Platform`) & Multi-Tenant GPU Partitioning Selection Guide: Siloed Departmental Clusters vs. GPU Time-Slicing (`OOM` Interference Anti-Pattern) vs. Static `NVIDIA MIG` vs. 3-Layer Internal `MaaS` (`API Virtualization Pool + Hardware MIG + Kueue Fair-Sharing`)

> **Language / 语言**: **English (Current)** | [中文](./enterprise-internal-maas-platform-multi-tenant-gpu-sharing-selection-faq.md)  
> **Scope**: End-to-end production reference architecture and multi-tenant GPU governance guide for Cloud & AI Solution Architects, Internal AI Platform Engineering Leads, and Kubernetes Infrastructure Architects.  
> **Provenance**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`). Deterministic evaluation model sealed in [`experiments/enterprise_internal_maas_platform_multi_tenant_gpu_sharing_selection/results.json`](../../experiments/enterprise_internal_maas_platform_multi_tenant_gpu_sharing_selection/results.json).

![Enterprise Multi-Team Internal MaaS Platform & GPU Sharing Summary](../../assets/svg/enterprise-internal-maas-platform-multi-tenant-gpu-sharing-selection-faq-summary.svg)

---

## ① 📌 Executive Summary (3-Minute Architect Verdict)

When **25 internal engineering and business-unit squads** build AI features simultaneously, handing out dedicated GPU nodes to each team creates severe GPU scarcity and operational sprawl:
1. **Anti-Pattern #1: Siloed Departmental GPU Clusters (`1 Team = 1 Isolated Cluster`, Tier 1, `22.0%` Utilization, `80x H100 GPUs`, `$310,000/month`)**: Because 85% of internal squads need the exact same foundation models (`Llama-3.3-70B / Qwen2.5-72B`, `Coder-32B`, `BGE-M3` embeddings, and rerankers), giving each team its own GPU nodes loads 25 redundant copies of 72GB weights into HBM memory (`1,800 GB` of duplicated weights!) while onboarding a new project takes **14 days** of ticket approvals;
2. **Anti-Pattern #2: Un-Isolated GPU Time-Slicing or `CUDA MPS` Across Multi-Team Containers (Tier 2, `31.5%` Noisy-Neighbor `CUDA OOM` Crash Rate)**: Software time-slicing shares a single GPU memory address space and fault domain—when Team A's OCR pod spikes VRAM or triggers a page fault, Team B's and Team C's production pods on the same GPU crash immediately;
3. **Limitation #3: Static Hardware `NVIDIA MIG` Slices Only (Tier 3, `54.0%` Utilization)**: Small MIG slices (`1g.10gb` / `2g.20gb`) cannot fit a 70B model, and statically assigned MIG slices sit idle when their owning department is off-peak.

Evaluated across **25 internal teams (`85%` shared foundation model traffic + `15%` custom small models and LoRA fine-tuning jobs)**, the four internal MaaS architectures compare as follows:

| Architectural Tier | Core Multi-Tenant Sharing Stack | Cluster GPU Utilization | Noisy-Neighbor `OOM` / Lag | New Team Onboarding | Required `H100` Fleet | Monthly Platform TCO | Production Verdict |
|---|---|---|---|---|---|---|---|
| **🟥 Tier 1: Siloed Departmental GPU Nodes** | 25 isolated clusters loading duplicate 70B weights | `22.0%` ✕ *(78% idle capacity)* | `0.0%` | `14.0 days` ✕ *(slow procurement)* | `80x H100` ✕ | `$310,000 / mo` ✕ | ⛔ #1 cause of enterprise GPU shortages |
| **🟨 Tier 2: Un-Isolated Time-Slicing / `MPS`** | Multi-tenant pods sharing one GPU address space | `68.0%` | `31.5%` ✕ *(cross-team OOM crashes)* | `3.0 days` | `40x H100` | `$178,000 / mo` ✕ *(incl. outages)* | ⛔ Unsafe for multi-tenant production serving |
| **🟦 Tier 3: Static Hardware `NVIDIA MIG` Only** | Fixed `1g.10gb / 3g.40gb` slices per team | `54.0%` | `0.0%` *(silicon isolation)* | `2.0 days` | `48x H100` | `$164,000 / mo` | ⚠️ Great for small models, but cannot pool 70B |
| **🟩 Tier 4: 3-Layer Internal `MaaS` (`API Pool + MIG + Kueue`)** | **L1 Shared Multi-Tenant API Pool (`85%` load) + L2 Silicon `NVIDIA MIG` (`<=8B`) + L3 `Kueue` Preemptible Cohort Borrowing** | **`86.5%` 🏆** *(3.9x higher util)* | **`0.0%` 🏆** *(Chunked Prefill + MIG)* | **`30 mins (0.02d)` 🏆** *(self-service key)* | **`24x H100` 🏆** *(-70% GPUs)* | **`$82,000 / mo` 🏆** *(-73.5%)* | 🌟 **Recommended Enterprise Internal MaaS Standard** |

> 💡 **Golden Rule for Enterprise Internal MaaS Platform Architects**: **"Share at the highest possible abstraction layer! (1) Serve the `85%` common foundation model demand (`70B`, `Coder-32B`, `Embedding`, `Reranker`) from a centralized Multi-Tenant API Pool (`vLLM / SGLang` with `Chunked Prefill` and `Multi-LoRA`) where teams onboard in `30 minutes` via self-service Virtual API Keys; (2) isolate team-specific custom small models (`<= 8B`, OCR, Whisper) inside hardware `NVIDIA MIG` slices (`1g.10gb` / `2g.20gb`) so a memory leak never crashes a neighbor (`0.0%` OOM); and (3) schedule fine-tuning and batch evals via `Kubernetes Kueue` Cohort Fair-Sharing so teams can borrow idle GPUs and yield them back in `30 seconds`."**

---

## ② 🎯 Customer & Architect Deep-Dive Q&A (3 Internal MaaS Consolidation, MIG vs. Time-Slicing & Quota Borrowing Pain Points)

### Q1: Why does consolidating 25 teams onto a Shared API-Level `MaaS` Pool cut GPU requirements from `80x H100` down to `24x H100`, and how do we prevent a single team's `128K-token` prompt from stalling everyone else?
- **Eliminating Duplicate Model Weights + `Chunked Prefill` QoS**:
  - Loading one shared `70B FP8` model pool (plus hot-swapped 160MB `LoRA` adapters) instead of 25 separate departmental deployments reclaims over `1,500 GB` of wasted HBM3 VRAM and multiplexes bursty team traffic into one high-utilization continuous batch (`86.5%` GPU utilization).
  - Combining AI Gateway per-team **`GCRA` token-bucket rate limits** with **`vLLM --enable-chunked-prefill`** slices any long 128K-token prompt into small `2,048-token` chunks interleaved with other tenants' decode steps (`0.0%` head-of-line blocking).

### Q2: When teams deploy custom small containers (OCR, audio, 7B classifiers), why must we use hardware `NVIDIA MIG` instead of Kubernetes GPU Time-Slicing?
- **Silicon Memory-Controller & Fault Isolation**:
  - Kubernetes GPU Time-Slicing and CUDA MPS multiplex threads onto the same GPU without isolating HBM memory or SMs; an OOM or kernel fault in one container kills every container on the GPU (`31.5%` interference).
  - **Hardware `NVIDIA MIG`** partitions the H100 into physically isolated GPU instances (`1g.10gb`, `2g.20gb`, `3g.40gb`), each with dedicated HBM memory channels, L2 cache banks, and compute SMs (`0.0%` cross-tenant OOM or latency bleed).

### Q3: How does `Kubernetes Kueue` let departments share idle training/batch GPUs without losing guaranteed access to the hardware they funded?
- **Cohort Fair-Sharing with 30-Second Graceful Preemption**:
  - Each funding business unit gets a guaranteed `nominalQuota` in its `ClusterQueue` within a shared `Cohort`. When Retail BU's GPUs are idle, Risk BU's `P2` fine-tuning jobs borrow those GPUs automatically; the moment Retail BU submits a `P0` workload, `Kueue` checkpoints and preempts the borrowed job in `< 30 seconds`.

---

## ③ 🧭 Visual Architecture Diagrams (Decision Tree + 3-Layer Internal MaaS Topology + Sequence Flow)

### 3.1 Enterprise Internal MaaS & Multi-Tenant GPU Partitioning Decision Tree

```mermaid
flowchart TD
    A["Enterprise Multi-Team AI Platform & GPU Sharing Selection<br/>(25 Internal Squads + Strict FinOps & SLA Isolation)"] --> B{"What workload is the internal team onboarding?"}
    B -- "1. Foundation LLM / Coding / Embedding / Reranker (85% of teams)" --> C["🌐 Layer 1: Shared Multi-Tenant MaaS API Pool<br/>Self-Service Virtual API Key (30m) + Chunked Prefill + Multi-LoRA<br/>(Never allocate dedicated GPU nodes for standard 70B calls!)"]
    B -- "2. Custom Small Model Container (<=8B / OCR / Whisper, 10%)" --> D["🧩 Layer 2: Hardware NVIDIA MIG Partition Pool<br/>Assign 1g.10gb / 2g.20gb / 3g.40gb silicon-isolated slices<br/>(0.0% cross-tenant OOM; ban software Time-Slicing in prod!)"]
    B -- "3. LoRA Fine-Tuning (SFT) / Bulk Evaluation Jobs (5%)" --> E["⚖️ Layer 3: Kubernetes Kueue Preemptible Cohort Pool 🏆<br/>Borrows idle quota across departments (86.5% GPU util)<br/>with 30s checkpoint-and-preempt return when owner needs GPUs!"]
    style C fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#f8fafc
    style D fill:#1e3a8a,stroke:#3b82f6,stroke-width:2px,color:#f8fafc
    style E fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#f8fafc
```

### 3.2 Tier 4 Three-Layer Enterprise Internal `MaaS` Platform (`API Pool + NVIDIA MIG + Kueue`) Architecture

```mermaid
flowchart LR
    subgraph Portal["1. Self-Service Internal Developer Portal (30m Onboarding)"]
        Teams["25 Internal Product Squads"] --> DevPortal["🔑 Internal MaaS Portal (Backstage / APIM / LiteLLM)<br/>Issues Virtual API Keys with Cost-Center Tags,<br/>GCRA Token Quotas, Guardrails & OTel Tracing"]
    end

    subgraph Layers["2. 3-Layer Multi-Tenant GPU Fleet (Consolidated to 24x H100)"]
        DevPortal -->|85% Foundation & LoRA Calls| L1["🚀 Layer 1: Shared Foundation API Pool (vLLM / SGLang)<br/>70B + Coder-32B + BGE-M3 + Dynamic Multi-LoRA<br/>Chunked Prefill prevents long-prompt stalls"]
        DevPortal -->|10% Custom Small Containers| L2["🛡️ Layer 2: Silicon NVIDIA MIG Pool<br/>1g.10gb / 2g.20gb / 3g.40gb hardware slices<br/>Dedicated HBM & SMs (0.0% OOM bleed)"]
        DevPortal -->|5% SFT & Eval Jobs| L3["🔄 Layer 3: Kubernetes Kueue Cohort Pool<br/>Cross-BU idle GPU borrowing + 30s P0 preemption"]
    end
```

### 3.3 End-to-End Sequence: 30-Minute Self-Service Onboarding & `Kueue` Cross-Departmental GPU Borrowing

```mermaid
sequenceDiagram
    autonumber
    participant TeamA as Retail Squad (Owns 8-GPU Quota)
    participant TeamB as Risk Squad (New Project)
    participant Portal as Self-Service MaaS Portal & Gateway
    participant Pool as Layer 1 Shared 70B MaaS Pool
    participant Kueue as Layer 3 K8s Kueue Cohort Scheduler

    TeamB->>Portal: Request access to `Llama-3.3-70B` and `BGE-M3` in Developer Portal
    Portal-->>TeamB: Issue scoped Virtual API Key (`10K TPM`, `bu=risk`) in 30 minutes (vs 14-day hardware ticket!)
    TeamB->>Pool: Serve traffic immediately from shared 70B pool (Chunked Prefill guarantees 0.0% interference)
    Note over TeamA,Kueue: Retail Squad's 4 training/batch GPUs are idle at 14:00
    TeamB->>Kueue: Submit `P2` LoRA fine-tuning job
    Kueue->>Kueue: Lend Retail Squad's 4 idle GPUs to Risk Squad automatically (cluster hits 86.5% utilization!)
    TeamA->>Kueue: Submit urgent `P0` Retail job at 16:00
    Kueue->>TeamB: Checkpoint & preempt borrowed `P2` pod in 30s and return all 4 GPUs to Retail Squad!
```

---

## ④ 📊 4-Tier Architecture Decision Matrix

| Evaluation Dimension | 🟥 Tier 1: Siloed Team Nodes | 🟨 Tier 2: GPU Time-Slicing / `MPS` | 🟦 Tier 3: Static Hardware `MIG` Only | 🟩 Tier 4: 3-Layer Internal `MaaS` (`API + MIG + Kueue`) |
|---|---|---|---|---|
| **Cluster GPU Utilization** | `22.0%` ✕ (78% idle waste) | `68.0%` | `54.0%` | 🏆 **`86.5%` (shared API pool + Kueue borrowing)** |
| **Noisy-Neighbor `OOM` / Lag** | `0.0%` | `31.5%` ✕ (shared fault domain) | `0.0%` | 🏆 **`0.0%` (Chunked Prefill + silicon MIG slices)** |
| **New Project Onboarding Time** | `14.0 days` ✕ | `3.0 days` | `2.0 days` | 🏆 **`30 mins (0.02 days)` (self-service Virtual Key)** |
| **Required `H100` Fleet (`25 Teams`)** | `80x H100` ✕ | `40x H100` | `48x H100` | 🏆 **`24x H100` (-70% GPU footprint)** |
| **25-Team Monthly Platform TCO** | `$310,000 / mo` | `$178,000 / mo` (incl. crashes) | `$164,000 / mo` | **`$82,000 / mo` 🏆 (-73.5%, saves $2.73M/yr)** |

---

## ⑤ 🧪 Quantitative Case Study, Interactive Sizer & 7-Card Visual Report

- **Interactive HTML Sizer**: 👉 [Open Enterprise Multi-Team Shared Internal MaaS Platform & GPU Sizer](../reports/enterprise-internal-maas-platform-multi-tenant-gpu-sharing-sizer.html)
- **7-Card 4:5 Visual Benchmark Report**: 👉 [Open 7-Card Visual Benchmark Deck](../reports/enterprise-internal-maas-platform-multi-tenant-gpu-sharing-selection-cards.html)
- **Deterministic Verification**:
  ```bash
  python experiments/enterprise_internal_maas_platform_multi_tenant_gpu_sharing_selection/run.py --check
  python -m unittest tests/test_enterprise_internal_maas_platform_multi_tenant_gpu_sharing_selection.py
  ```

---

## ⑥ 🛠️ Production Readiness & Architecture Review Checklist

- [ ] **Consolidate Common Foundation Models (`70B`, `32B`, `Embedding`, `Reranker`) into a Shared Multi-Tenant `MaaS` API Pool**: Onboard internal squads via self-service AI Gateway Virtual Keys rather than provisioning dedicated GPU nodes per squad.
- [ ] **Enable `Chunked Prefill` (`--enable-chunked-prefill`) and Per-Key `GCRA` Token-Bucket Quotas on All Shared Serving Pools**: Prevent long-context requests from causing head-of-line decode stalls for co-tenants.
- [ ] **Enforce Hardware `NVIDIA MIG` (`1g.10gb`, `2g.20gb`, `3g.40gb`) for Co-Located Custom Small Models**: Never use un-isolated GPU Time-Slicing for multi-tenant production serving.
- [ ] **Deploy `Kubernetes Kueue` with Cohort Quota Borrowing and Graceful Checkpoint Preemption for Fine-Tuning and Batch Evaluation Jobs**.
- [ ] **Automate GPU Hardware Fault Remediation (`DCGM + Node Problem Detector`)**: Automatically cordon and drain nodes exhibiting `Xid 79` or double-bit ECC memory errors.
