# Prefill-Decode (`PD`) Disaggregation Deep Dive & Decision Boundary: Why Prefill & Decode Collide, When to Keep Colocated (`Chunked Prefill`) vs. Disaggregate, `KV Cache` Transfer Bandwidth Math & Engine Selection (`Dynamo / Mooncake / SGLang / vLLM`) Guide

> **Target Audience**: Cloud & AI Infrastructure Solution Architects, LLM Serving Platform Engineers, and AI Network Architects deciding between Colocated Continuous Batching (`+ Chunked Prefill`) and Prefill-Decode (`PD`) Disaggregation.
> **Core Question**: Why do **Prefill** (compute-bound, $\text{AI} > 300\text{ FLOPs/Byte}$) and **Decode** (memory-bandwidth-bound, $\text{AI} \approx 1\text{–}16\text{ FLOPs/Byte}$) fundamentally interfere when co-located on the same GPUs? When a `16,384`-token RAG/Agent request enters a GPU streaming 64 active Decode users, why does P99 `TPOT` spike from `16ms` to **`148ms` (or `58.5ms` even with `Chunked Prefill`, breaking a `<=25ms` SLA)**? **What are the exact quantitative boundaries (GPU count, prompt length, `I:O` token ratio, SLA, and network bandwidth) for keeping Colocated + Chunked Prefill vs. switching to PD Disaggregation?** How do you hand-calculate cross-node `KV Cache` transfer time across `NVLink`, `400G RoCEv2/IB`, and `TCP/IP`, and how do `Mooncake`, `NVIDIA Dynamo (NIXL)`, `SGLang`, and `vLLM V1` compare?

![PD Disaggregation vs. Colocated Chunked Prefill Boundary Summary](../../assets/svg/prefill-decode-disaggregation-vs-colocated-chunked-prefill-boundary-selection-faq-summary.svg)

---

## 1. 📌 Executive Summary (3-Minute Verdict)

**The First-Principles Physics of PD Interference**: Prefill is **Compute-Bound** (needs small batch size and high Tensor Parallelism `TP` to saturate TensorCores and minimize `TTFT`), whereas Decode is **HBM-Bandwidth-Bound** (needs large batch size `B=64–256` and high Expert/Data Parallelism `EP/DP` so dozens of tokens amortize reading the model weights from HBM once per step). Co-locating them on the same GPUs forces a **head-on collision between freight trains (16K Prefills) and Formula-1 cars (16ms Decode steps)**:

| Serving Architecture Tier | Core Scheduling Mechanism | `16K` Prompt `P99 TTFT` | `16K` Prompt `P99 TPOT` (`<=25ms` SLA) | SLA-Compliant `Goodput` (`tok/s/GPU`) | Cost / 1M SLA Tokens (`$/1M`) | Architect Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Tier 1: Naive Colocated Continuous Batching (`Prefill-First`, No Chunking)** | Arriving Prefills preempt active Decode batches | `620 ms` | **`148.0 ms` (`5.9x` over SLA!)** | `520 tok/s/GPU` | `$0.92 / 1M` | ❌ **Reject**: Every long prompt freezes streaming output for 140ms+ |
| **Tier 2: Colocated + `Chunked Prefill` (`chunk_size = 1024 / 2048`)** | Splits 16K prompt into 16 x 1K chunks piggybacked on Decode | `940 ms` | `58.5 ms` (exceeds `25ms` SLA) | `890 tok/s/GPU` | `$0.64 / 1M` | ✅ **Best for Small Pools / Short Prompts**: Ideal when `<=8 GPUs`, `<2K` prompt, or no RDMA |
| **Tier 3: PD Disaggregation Over Standard `TCP/IP` (No RDMA)** | Separates P/D, but copies `2.68 GB` KV over kernel TCP | **`1,120 ms` (`+480ms` KV stall!)** | `21.0 ms` | `1,050 tok/s/GPU` | `$0.55 / 1M` | ❌ **Reject**: Kernel TCP KV transfer bottleneck ruins `TTFT` |
| **Tier 4: PD Disaggregation + `400G GPUDirect RDMA` Layer-Wise KV Push (`Dynamo / Mooncake / SGLang`)** | Dedicated P & D pools + Layer-$L$ RDMA push overlapped with Layer-$L+1$ | **`485 ms`** | **`16.4 ms` (`3.57x` lower!)** | **`1,940 tok/s/GPU` (`2.18x` higher)** | **`$0.34 / 1M` (`-46.9%`)** | 🏆 **Production Gold Standard (`>=16 GPUs`, `>=4K` Prompt, `I:O >= 8:1`)** |

---

## 2. 🎯 Customer & Architect Critical Q&A (Physics, Boundaries & Hand-Calculations)

### Q1: Why does `Chunked Prefill` (Tier 2) still lose to PD Disaggregation by `2.18x` in SLA-compliant Goodput on `16K` prompts?
* **Three Structural Limitations of Co-Located Chunked Prefill**:
  1. **Quadratic KV Cache Reload Tax from HBM**: Splitting a `16K` prompt into 16 chunks of `1K` tokens means Chunk 2 reloads Chunk 1's KV from HBM, Chunk 3 reloads Chunks 1–2, ..., and Chunk 16 reloads Chunks 1–15—reading historical prefix KV $\frac{16 \times 17}{2} = 136$ chunk-times from HBM!
  2. **Mismatched Optimal Parallelism (`TP` vs. `DP+EP`)**: Prefill wants high `TP=8` to minimize `TTFT`, while Decode on large MoE models (`DeepSeek-671B`) wants wide `Attention DP + Expert EP=32/64` with huge batch sizes to maximize HBM `MBU`. Co-located instances cannot run two different parallelism topologies simultaneously.
  3. **Inability to Mix Heterogeneous GPU SKUs**: PD Disaggregation lets you run compute-dense **Prefill on `H100 / B200`** while running memory-capacity/bandwidth-bound **Decode on `H200 (141GB) / MI300X (192GB)`**.

### Q2: [The Architect's Decision Matrix] When should you keep `Colocated + Chunked Prefill` vs. mandate `PD Disaggregation`?

| Decision Dimension | ✅ Keep **Colocated + Chunked Prefill** (Simpler Ops) | 🚀 Mandate **PD Disaggregation** (`Dynamo / Mooncake / SGLang`) | Architectural Rationale |
| :--- | :--- | :--- | :--- |
| **1. Fleet Size (GPUs per Model)** | **`<= 8 GPUs`** (single node or small replica) | **`>= 16 GPUs`** (at least `1–2` Prefill nodes + `2–4` Decode nodes) | Splitting `<8` GPUs into separate P and D pools causes queueing starvation |
| **2. Average Input Prompt ($S_{\text{in}}$)** | **Short Prompts ($S_{\text{in}} < 2,048\text{ tokens}$)** | **Medium-to-Long Context ($S_{\text{in}} \ge 4,096\text{–}128,000\text{ tokens}$)** | `<2K` Prefills finish in `<30ms` with minimal Decode impact |
| **3. Input-to-Output Ratio (`I:O`)** | **Low `I:O < 4 : 1`** (creative writing, long-CoT reasoning output) | **High `I:O >= 8 : 1` to `50 : 1`** (Enterprise RAG, repo-level coding, multi-turn agents) | High `I:O` floods the cluster with heavy Prefill compute bursts |
| **4. Tail-Latency SLA (`P99 TPOT`)** | Relaxed (`P99 TPOT <= 60–100ms` or async batch) | **Strict Interactive SLA (`P99 TPOT <= 20–25ms` & `P99 TTFT <= 800ms`)** | Physical isolation guarantees zero Prefill jitter on Decode streams |
| **5. Inter-Node Network Fabric** | **Standard `10G/25G/100G TCP/IP` only (no RDMA)** | **`100G–800G GPUDirect RDMA (RoCEv2 / IB)`, Rack `NVLink`, or `TPU ICI`** | Un-accelerated TCP takes `480ms+` to copy multi-GB KV caches |

### Q3: [Step-by-Step Math] How do you hand-calculate `KV Cache` transfer payload and network latency ($BW_{\text{KV}}$)?
1. **Step 1: Compute Single-Request KV Payload ($M_{\text{KV\_req}}$)**:
   $$M_{\text{KV\_req}} = 2 \times L \times n_{\text{kv\_heads}} \times d_{\text{head}} \times S_{\text{in}} \times \text{Bytes}_{\text{elem}}$$
   * **Example A (`Llama-3-70B` GQA: $L=80, n_{\text{kv}}=8, d_h=128, S_{\text{in}}=16,384$)**:
     * In **`BF16` (2B)**: $327.68\text{ KB/tok} \times 16,384 = \mathbf{5.37\text{ GB}}$.
     * In **`FP8` (1B)**: $163.84\text{ KB/tok} \times 16,384 = \mathbf{2.68\text{ GB}}$.
   * **Example B (`DeepSeek-V3/R1 671B` MLA: $L=61$, compressed latent dimension $= 576$, `FP8` 1B)**:
     * Only $\mathbf{35.1\text{ KB/tok}}$ (`FP8`) or $\mathbf{70.3\text{ KB/tok}}$ (`BF16`) — **`16K` prompt is just `0.575 GB` (`FP8`) / `1.15 GB` (`BF16`)**, a **`4.6x` reduction vs. GQA** and **`37x` vs. MHA**!
2. **Step 2: Compute Raw vs. Layer-Overlapped Transfer Time ($T_{\text{KV}} = \frac{M_{\text{KV\_req}}}{\text{BW}_{\text{eff}}}$)**:
   * Over **`400G RoCEv2 / IB GPUDirect RDMA` (`~45 GB/s` effective)**: $2.68\text{ GB} / 45\text{ GB/s} = \mathbf{59.6\text{ ms}}$ raw; with **Layer-Wise Pipelined Push** (streaming Layer $L$'s KV while computing Layer $L+1$), `75%` of transfer hides behind Prefill compute, leaving just **`14.9 ms`** exposed overhead!

---

## 3. 🧭 Visual Diagrams (Mermaid Pipeline Comparison + Decision Tree + Sequence Diagram)

### 3.1 Colocated Interference vs. Layer-Overlapped RDMA PD Disaggregation

```mermaid
flowchart TB
    subgraph Coloc["⛔ Colocated Mode: 16K Prefill Burst Freezes 64 Active Decode Streams for 60-148ms"]
        direction LR
        D1["Decode Step #1\n(16ms normal)"] --> D2["Decode Step #2\n(16ms normal)"]
        D2 --> P_IN["🚨 16K Prompt Arrives!\nPrefill Hijacks TensorCores for 140ms\n(All 64 streaming users freeze!)"]
        P_IN --> D3["Decode Step #3\n(P99 TPOT spikes to 148ms!)"]
    end

    subgraph Disagg["🏆 PD Disaggregated Mode: Isolated Pools + Layer-Wise 400G GPUDirect RDMA KV Push"]
        direction LR
        PP["Prefill Pool (TP=8 Compute-Optimized)\nProcesses 16K Prompt at Full Speed (470ms)"] ==>|"Streams Layer L KV via 400G RDMA\nwhile computing Layer L+1 (14.9ms net)"| DP["Decode Pool (Large Batch + EP/DP)\nUninterrupted 16.4ms P99 TPOT!"]
    end
```

### 3.2 Colocated vs. PD Disaggregation Architecture Decision Tree

```mermaid
flowchart TD
    A["Enterprise Online LLM Serving Architecture"] --> B{"Is the model pool >= 16 GPUs/TPUs?"}
    B -- "No (<=8 GPUs)" --> C["✅ Keep Colocated + Chunked Prefill\n(vLLM / SGLang chunked_prefill_size=1024)"]
    B -- "Yes (>=16 GPUs)" --> D{"Prompt Length & Input:Output (I:O) Ratio?"}
    D -- "Short Prompts (<2K) & I:O < 4:1" --> C
    D -- "Long Context (>=4K) or I:O >= 8:1 (RAG / Agent / Coding)" --> E{"Is 100G-800G GPUDirect RDMA / NVLink available (or MLA model)?"}
    E -- "No RDMA & Large GQA/MHA KV" --> F["⚠️ Keep Colocated + Chunked Prefill + FP8 KV\n(Avoids 480ms+ kernel TCP KV copy stall)"]
    E -- "Yes (400G RDMA / NVLink / ICI or DeepSeek MLA)" --> G["🏆 Deploy Tier 4 PD Disaggregation (Dynamo / Mooncake / SGLang)\nDynamically auto-scale xP : yD ratio (e.g., 3P : 5D)"]
```

### 3.3 End-to-End Layer-Wise Zero-Copy `GPUDirect RDMA` KV Transfer Sequence

```mermaid
sequenceDiagram
    autonumber
    participant Client as User / Agent
    participant Router as KV-Aware Router (Dynamo / Mooncake)
    participant PNode as Prefill Worker (TP=8)
    participant RDMA as 400G ConnectX-7 (GPUDirect RDMA)
    participant DNode as Decode Worker (High-Batch MBU Pool)

    Client->>Router: Sends `16,384-token` RAG prompt
    Router->>Router: Matches `6,000-token` cached prefix on `PNode #2` & pre-allocates paged KV slots on `DNode #5`
    Router->>PNode: Dispatches prompt + `DNode #5` remote HBM RDMA descriptors (`rkey / vaddr`)
    loop Layers L = 1 .. 80 Pipelined Compute & RDMA Write
        PNode->>PNode: Computes Layer $L$ K, V tensors on TensorCores
        PNode->>RDMA: Posts async `IBV_WR_RDMA_WRITE` from PNode HBM directly to DNode HBM (overlapped with Layer $L+1$!)
    end
    PNode-->>Client: Emits first token (`P99 TTFT = 485ms`)
    RDMA->>DNode: Signals Layer-80 KV transfer complete (`+14.9ms` exposed)
    loop Tokens 2 .. 512
        DNode-->>Client: Streams uninterrupted tokens at `P99 TPOT = 16.4ms` (`1,940 tok/s/GPU` Goodput)
    end
```

---

## 4. 📊 4-Tier Architecture Decision Matrix

| Evaluation Dimension | Tier 1: Naive Colocated (`Prefill-First`) | Tier 2: Colocated + `Chunked Prefill` (`1024`) | Tier 3: PD Disaggregation Over `TCP/IP` | Tier 4: PD Disaggregation + `400G GPUDirect RDMA` Layer-Wise Push (Recommended) |
| :--- | :--- | :--- | :--- | :--- |
| **`16K` Prompt `P99 Decode TPOT`** | **`148.0 ms` (5.9x SLA breach)** | `58.5 ms` (exceeds `25ms` SLA) | `21.0 ms` | **`16.4 ms` (`3.57x` lower than Chunked Colocated!)** |
| **`16K` Prompt `P99 Prefill TTFT`** | `620.0 ms` | `940.0 ms` (16-step chunk delay) | **`1,120.0 ms` (`+480ms` TCP stall)** | **`485.0 ms` (dedicated `TP=8` + overlapped RDMA)** |
| **SLA-Compliant `Goodput` (`tok/s/GPU`)** | `520 tok/s/GPU` | `890 tok/s/GPU` | `1,050 tok/s/GPU` | **`1,940 tok/s/GPU` (`2.18x` higher than Tier 2)** |
| **Independent P/D Parallelism & SKUs** | ❌ Locked together | ❌ Locked together | ✅ Decoupled | **✅ High-`TP` Prefill + High-`EP/DP` Large-Batch Decode** |
| **Cost per 1M SLA Tokens (`$/1M`)** | `$0.92 / 1M` | `$0.64 / 1M` | `$0.55 / 1M` | **`$0.34 / 1M` (`-46.9%` savings vs. Tier 2)** |

---

## 5. 🧪 Quantitative Case Study, Interactive H5 Sizer & 7-Card Report

### 5.1 Baseline Workload (`Llama-3-70B-FP8` & `DeepSeek-671B` / `16K` Input + `512` Output)
* **Workload Profile**: Enterprise RAG and Agent workload with `16,384` average input tokens and `512` output tokens (`I:O = 32:1`, `P99 TTFT <= 800ms`, `P99 TPOT <= 25ms`).
* **Reproducible Model & Dataset**: See [`experiments/prefill_decode_disaggregation_vs_colocated_chunked_prefill_boundary_selection/run.py`](../../experiments/prefill_decode_disaggregation_vs_colocated_chunked_prefill_boundary_selection/run.py) and [`results.json`](../../experiments/prefill_decode_disaggregation_vs_colocated_chunked_prefill_boundary_selection/results.json) (`provenance: synthetic + anonymized_reported`, `independently_reproduced=false`).
* **Interactive H5 Sizer & KV Bandwidth Calculator**: 📊 [Open PD Disaggregation vs. Colocated Boundary & KV Bandwidth Sizer (`prefill-decode-disaggregation-vs-colocated-chunked-prefill-boundary-sizer.html`)](../reports/prefill-decode-disaggregation-vs-colocated-chunked-prefill-boundary-sizer.html)
* **7-Card 4:5 Benchmark Deck**: 🃏 [Open 7-Card Visual Report (`prefill-decode-disaggregation-vs-colocated-chunked-prefill-boundary-selection-cards.html`)](../reports/prefill-decode-disaggregation-vs-colocated-chunked-prefill-boundary-selection-cards.html)

```chart
{
  "data": {
    "values": [
      { "tier": "Tier 1: Naive Colocated", "p99_tpot_ms": 148.0, "p99_ttft_ms": 620.0, "goodput_tok_s": 520, "cost_per_1m_usd": 0.92 },
      { "tier": "Tier 2: Colocated + Chunked Prefill", "p99_tpot_ms": 58.5, "p99_ttft_ms": 940.0, "goodput_tok_s": 890, "cost_per_1m_usd": 0.64 },
      { "tier": "Tier 3: PD Split Over TCP/IP", "p99_tpot_ms": 21.0, "p99_ttft_ms": 1120.0, "goodput_tok_s": 1050, "cost_per_1m_usd": 0.55 },
      { "tier": "Tier 4: 400G RDMA Layer-Wise PD Split", "p99_tpot_ms": 16.4, "p99_ttft_ms": 485.0, "goodput_tok_s": 1940, "cost_per_1m_usd": 0.34 }
    ]
  },
  "semantic_types": {
    "tier": "Category",
    "p99_tpot_ms": "Quantity",
    "p99_ttft_ms": "Quantity",
    "goodput_tok_s": "Quantity",
    "cost_per_1m_usd": "Price"
  },
  "chart_spec": {
    "chartType": "Bar Chart",
    "title": "400G RDMA PD Disaggregation Cuts P99 TPOT 3.57x (16.4ms) and Lifts SLA Goodput 2.18x (1,940 tok/s/GPU)",
    "subtitle": "SLA-compliant goodput (tokens/sec/GPU) across colocated and disaggregated serving tiers on 16K prompts",
    "encodings": {
      "x": "tier",
      "y": "goodput_tok_s",
      "color": "tier"
    }
  }
}
```

---

## 6. 🛠️ Architecture Review & Production Readiness Checklist

- [ ] **5-Axis Boundary Gate (`>=16 GPUs`, `>=4K` Prompt, `I:O >= 8:1`, `TPOT <= 25ms`, `RDMA / MLA`)**: Has the architecture team verified that the deployment meets the threshold for PD Disaggregation rather than prematurely splitting a `<8-GPU` or non-RDMA pool?
- [ ] **Layer-Wise Overlapped `GPUDirect RDMA` KV Push (`NIXL / Mooncake`)**: Is KV transfer pipelined layer-by-layer (`FP8` quantized) over `GPUDirect RDMA` so `<20%` of transfer time is exposed on `TTFT`?
- [ ] **Dynamic `xP : yD` Pool Ratio Autoscaler**: Does the orchestrator dynamically adjust the ratio of Prefill to Decode instances as live `I:O` token ratios shift across business hours?
- [ ] **Pre-Allocated Decode Paged KV Descriptors**: Are destination HBM page-table addresses allocated on the Decode worker *before* Prefill starts so Prefill workers never block waiting for Decode memory?
- [ ] **Prefix-Aware Global KV Cache Integration**: Are Prefill workers integrated with a distributed DRAM/NVMe prefix cache (`Mooncake / RadixAttention`) so cached prefixes skip both recomputation and redundant cross-node transfers?
