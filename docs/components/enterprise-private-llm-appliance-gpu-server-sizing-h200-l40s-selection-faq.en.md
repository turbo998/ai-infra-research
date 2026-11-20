# Enterprise On-Premises Private LLM Appliance & GPU Server Sizing Guide (`4x L40S` vs. `8x H100` vs. `8x H200`): Exact 32K-Context VRAM/KV-Cache Formulas, Legacy `6kW` Rack Power Boundaries & Single-Box `MIG` Isolation

> **Language / 语言**: **English (Current)** | [中文](./enterprise-private-llm-appliance-gpu-server-sizing-h200-l40s-selection-faq.md)  
> **Scope**: End-to-end hardware sizing and private AI appliance architecture guide for Cloud & AI Solution Architects, Enterprise Data Center Infrastructure Leads, and Regulated-Industry CTOs purchasing on-premises GPU servers for `30–250+` concurrent users.  
> **Provenance**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`). Deterministic evaluation model sealed in [`experiments/enterprise_private_llm_appliance_gpu_server_sizing_h200_l40s_selection/results.json`](../../experiments/enterprise_private_llm_appliance_gpu_server_sizing_h200_l40s_selection/results.json).

![On-Premises Private LLM Appliance & GPU Server Sizing Summary](../../assets/svg/enterprise-private-llm-appliance-gpu-server-sizing-h200-l40s-selection-faq-summary.svg)

---

## ① 📌 Executive Summary (3-Minute Architect Verdict)

When banks, hospitals, defense contractors, and manufacturing plants purchase on-premises GPU servers for private LLM and RAG deployment (**`Qwen2.5-72B / Llama-3.3-70B` with `32K` context + `BGE-M3` Embedding/Reranker + PDF OCR**), three hardware-sizing blunders routinely derail projects:
1. **Trap #1: Running Unquantized `BF16` with `TP=8` Across 8 PCIe `L40S (384GB)` Cards Without NVSwitch (Tier 1, `84.0ms` Decode TPOT, `OOM` at Just `14` Concurrent 32K Users)**: Cross-socket PCIe `AllReduce` across dual-CPU UPI interconnects cripples token generation speed, while uncompressed `FP16 KV Cache` consumes **`10.24 GB` per 32K user**!
2. **Trap #2: Dropping a `10.2 kW` `8x H100 SXM` Server into a Legacy `6.0 kW` Air-Cooled Enterprise Server Room (Tier 2, `$456,000` Total Cost Including a `$140K` Facility Power/Cooling Retrofit)**: Ordering an SXM server without checking rack PDU amperage and cooling CFM trips circuit breakers on Day 1;
3. **Trap #3: Co-Locating `72B LLM + Embedding + Reranker + PDF OCR` on Unpartitioned GPUs (Tier 3, Sporadic `CUDA OOM` Crashes)**: Whenever a user uploads a 100-page scanned PDF, the OCR batch spike steals VRAM from `vLLM` and crashes active chat sessions.

Evaluated for an **on-premises `72B` LLM + `32K` RAG context + full retrieval/OCR stack**, the four private appliance configurations compare as follows:

| Architectural Tier | Server SKU, Precision & Parallelism | Max `32K` Users / Node | Decode `TPOT P95` | Node Power Draw | 3-Yr Cost / Concurrent User | 3-Yr Appliance TCO | Production Verdict |
|---|---|---|---|---|---|---|---|
| **🟥 Tier 1: `8x L40S PCIe` (`BF16`, Cross-Socket `TP=8`)** | 144GB BF16 weights + FP16 KV across 8 PCIe cards | `14 users` ✕ *(KV exhaustion)* | `84.0 ms/tok` ✕ *(PCIe bottleneck)* | `5.2 kW` | `$11,400 / user` ✕ | `$342,000` | ⛔ Never run `TP=8` over PCIe without NVSwitch |
| **🟨 Tier 2: `8x H100 SXM` in Legacy `6kW` Room (`30 Users`)** | 10.2kW server forces $140K facility power/cooling retrofit | `95 users` | `18.5 ms/tok` | `10.2 kW` ✕ *(trips 6kW breaker)* | `$15,200 / user` ✕ *(at 30 users)* | `$456,000` ✕ *(incl. retrofit)* | ⛔ Mismatched to legacy `<6kW` air-cooled rooms |
| **🟦 Tier 3: `4x L40S PCIe` (`AWQ INT4` + `FP16 KV` Unisolated)** | INT4 weights, uncompressed FP16 KV, unisolated OCR | `28 users` | `46.0 ms/tok` | `3.8 kW` | `$6,900 / user` | `$207,000` | ⚠️ Lacks FP8 KV compression & MIG isolation |
| **🟩 Tier 4: Facility-Matched (`<=40 Users`: `4x L40S 3.6kW`; `50–160 Users`: `8x H200 1,128GB SXM` + `FP8 KV` + `MIG`)** | **`4x L40S (TP=4)` for `<6kW` racks; `8x H200 (1,128GB)` (`2x TP=4` + GPU-7 `MIG` for TEI/OCR) for core data centers** | **`160 users` 🏆** *(+68% vs H100)* | **`14.2 ms/tok` 🏆** *(4.8 TB/s HBM3e)* | **`3.6kW` / `10.2kW` 🏆** *(facility-matched)* | **`$2,350 / user` 🏆** *(-79.4% vs Tier 1)* | **`$376,000` 🏆** *(zero retrofit waste)* | 🌟 **Recommended Private AI Appliance Standard** |

> 💡 **Golden Rules for On-Premises AI Appliance Sizing**:
> 1. **Facility Power Boundary (`6.0 kW` Air-Cooled Ceiling)**: For legacy `< 6 kW` air-cooled server rooms (`<= 40` concurrent users), deploy a **`4U 4x NVIDIA L40S (192GB VRAM, 3.6 kW)`** running `72B AWQ/FP8 (TP=4)` + `FP8 KV Cache`—zero facility retrofit required!
> 2. **`H200 (141GB)` Strictly Dominates `H100 (80GB)` for Long-Context Private Appliances**: An `8x H200 (1,128GB HBM3e)` server costs only `~15%–18%` more than an `8x H100 (640GB)` server, but after subtracting `~160GB` for `72B` weights and auxiliary models, it provides **`+101.7%` more usable KV-Cache memory (`968GB` vs. `480GB`)** and **`4.8 TB/s` bandwidth (`160` concurrent `32K` users at `14.2ms` TPOT)**.
> 3. **Single-Box Partitioning (`2x TP=4` + Hardware `MIG`)**: Never bind all 8 GPUs into a single `TP=8` instance! Run **`TP=4` replicas on GPUs `0..3` and `4..6`** (so a single GPU fault leaves 50% capacity online) and partition **GPU `7` with Hardware `NVIDIA MIG`** for `TEI Embedding/Reranker` and `OCR`.

---

## ② 🎯 Customer & Architect Deep-Dive Q&A (3 Whiteboard VRAM Math, PCIe `TP=8` Trap & Single-Box `MIG` Pain Points)

### Q1: How do Solution Architects calculate the exact GPU VRAM required for `100` concurrent users at `32K` context on a whiteboard?
- **The 3-Part Private LLM VRAM Sizer Formula (`Weights + KV Cache + 12% Overhead`)**:
  1. **Model Weights (`72B`, 80 layers, GQA 8 KV heads, 128 head dim)**: `144 GB` in `BF16` $\rightarrow$ **`72 GB` in `FP8`** (or `38 GB` in `AWQ INT4`).
  2. **Per-Token & Per-User `32K` KV Cache**:
     $$\text{KV Bytes/Token} = 2 \times 80 \times 8 \times 128 \times \text{Bytes} = \begin{cases} 0.327\text{ MB/tok } (10.24\text{ GB per 32K user}) & \text{in FP16} \\ 0.164\text{ MB/tok } (5.12\text{ GB per 32K user}) & \text{in FP8} \end{cases}$$
     With **`FP8 KV Cache` + `50% Prefix Cache Sharing`** on system prompts and shared RAG documents, each active `32K` user consumes **`~2.5 GB` of net marginal KV VRAM**.
  3. **Usable KV Pool on `8x H200 (1,128GB)` vs. `8x H100 (640GB)`**: Subtracting `~160 GB` (weights + CUDA graphs + MIG slice), `8x H200` leaves **`968 GB` for KV Cache** (`160+` concurrent 32K users), whereas `8x H100` leaves **`480 GB`** (`95` users).

### Q2: Why does running `TP=8` across 8 PCIe `L40S` GPUs degrade token generation speed to `84ms/token`?
- **Avoiding Cross-Socket CPU UPI/QPI Interconnect Traversal**:
  - In a dual-socket PCIe server without NVSwitch, GPUs `0..3` sit on Socket 0 and GPUs `4..7` sit on Socket 1. `TP=8` forces 160 `AllReduce` barriers per token across the narrow inter-socket CPU link.
  - Restricting Tensor Parallelism to **`TP=4` within a single PCIe root complex** keeps all communication local to one PCIe switch, cutting decode latency from `84.0ms` down to **`24.5ms`** on `4x L40S` (and **`14.2ms`** over `900 GB/s` NVLink on `H200`).

### Q3: How do we safely run `72B LLM + Embedding + Reranker + PDF OCR` inside a single 8-GPU appliance without OOM crashes?
- **Hardware `NVIDIA MIG` Slicing on GPU 7 + Dedicated `TP=4` vLLM Pools on GPUs `0..6`**:
  - Partition `GPU 7 (141GB H200)` into hardware-isolated `MIG` instances (`2g.35gb` for `TEI Embedding`, `2g.35gb` for `TEI Reranker`, and `3g.70gb` for `Docling / VLM OCR`). Because `MIG` enforces hardware memory-controller isolation, a 500-page PDF upload can never touch a single byte of `vLLM`'s KV Cache on GPUs `0..6`!

---

## ③ 🧭 Visual Architecture Diagrams (Decision Tree + Single-Box `8x H200` `MIG` Layout + Concurrent Request Sequence)

### 3.1 On-Premises Private LLM Appliance & GPU Server Sizing Decision Tree

```mermaid
flowchart TD
    A["On-Premises Private LLM Appliance Sizing<br/>(72B Model + 32K RAG Context + Embedding/Rerank/OCR)"] --> B{"What is the data center rack power limit & target concurrency?"}
    B -- "Legacy air-cooled server room (<= 6kW/rack) or <= 40 concurrent users (~800 staff)" --> C["🟩 4U 4x NVIDIA L40S PCIe (192GB, 3.6kW) 🏆<br/>(Runs 72B AWQ/FP8 TP=4 + FP8 KV | Zero room retrofit needed!)"]
    B -- "Modern data center rack (>= 10–12kW/rack) & 50–160+ concurrent users (3,000+ staff)" --> D{"Which 8-GPU server architecture?"}
    D -- "8x L40S PCIe with cross-socket BF16 TP=8 (Tier 1)" --> E["⛔ 84ms Decode TPOT & OOM at 14 Users!<br/>Cross-socket PCIe AllReduce stalls every transformer layer"]
    D -- "8x NVIDIA H200 SXM (1,128GB HBM3e) + FP8 KV + GPU-7 MIG (Tier 4)" --> F["🟩 Tier 4: Single 8x H200 SXM (+101.7% Usable KV Pool vs H100) 🏆<br/>(160 Concurrent 32K Users | 14.2ms Decode TPOT | Single-Box 235B MoE Ready)"]
    style C fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#f8fafc
    style F fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#f8fafc
    style E fill:#7f1d1d,stroke:#ef4444,color:#fecaca
```

### 3.2 Tier 4 Single-Box `8x H200 (1,128GB)` All-in-One Appliance Partitioning Reference Architecture

```mermaid
flowchart LR
    subgraph Box["Single 8x NVIDIA H200 SXM5 Appliance (1,128GB HBM3e | 900GB/s NVLink | 10.2kW)"]
        subgraph PoolA["1. Primary LLM Pool A (GPUs 0..3 = 564GB)"]
            vLLMA["🚀 vLLM 72B FP8 Replica A (TP=4)<br/>72GB Weights + 450GB Usable FP8 KV Pool<br/>(Serves 85 Concurrent 32K Users)"]
        end
        subgraph PoolB["2. HA LLM Pool B (GPUs 4..6 = 423GB)"]
            vLLMB["🚀 vLLM 72B/32B FP8 Replica B<br/>(Serves 75 Concurrent 32K Users | Fault-Isolated)"]
        end
        subgraph MIG7["3. Hardware MIG Auxiliary GPU (GPU 7 = 141GB)"]
            Slice1["🔹 MIG 2g.35gb: HF TEI Embedding (BGE-M3)"]
            Slice2["🔹 MIG 2g.35gb: HF TEI Reranker (Top-20)"]
            Slice3["🔹 MIG 3g.70gb: Docling / Qwen2.5-VL PDF OCR"]
        end
    end
```

### 3.3 End-to-End Sequence: Simultaneous 200-Page PDF OCR Ingestion & `160`-User `32K` RAG Chat Without Contention

```mermaid
sequenceDiagram
    autonumber
    participant User as Enterprise Intranet Users (160 Concurrent)
    participant GW as Appliance Ingress Gateway
    participant MIG as GPU-7 (Hardware MIG Slices: TEI + OCR)
    participant LLM as GPUs 0..6 (vLLM 72B FP8 + FP8 KV Pool)

    par User A uploads a 200-page scanned contract PDF
        User->>GW: 1. Upload 200-page scanned PDF
        GW->>MIG: 2. Route to GPU-7 MIG Slice #3 (70GB hardware-fenced OCR)
        Note over MIG,LLM: Hardware MIG memory controllers block OCR spikes from touching GPUs 0..6!
    and Users B..N query 32K-context internal compliance RAG
        User->>GW: 3. Ask complex 32K-context policy comparison query
        GW->>MIG: 4. Route to GPU-7 MIG Slices #1/#2 (TEI Embed + Rerank in 28ms)
        GW->>LLM: 5. Dispatch to GPUs 0..3 (vLLM Replica A, TP=4) with FP8 KV Cache + Prefix Caching
        LLM-->>User: 6. Stream response at 14.2ms/token (70 tok/s) over 4.8 TB/s HBM3e!
    end
```

---

## ④ 📊 4-Tier Architecture Decision Matrix

| Evaluation Dimension | 🟥 Tier 1: `8x L40S` (`BF16 TP=8`) | 🟨 Tier 2: `8x H100` in `6kW` Room | 🟦 Tier 3: `4x L40S` (`FP16 KV`) | 🟩 Tier 4: `4x L40S` (`<6kW`) / `8x H200` (`FP8 KV + MIG`) |
|---|---|---|---|---|
| **Max `32K` Concurrent Users / Node** | `14 users` ✕ (FP16 KV OOM) | `95 users` | `28 users` | 🏆 **`160 users` (`8x H200 1,128GB` + `FP8 KV`)** |
| **Decode Latency (`TPOT P95`)** | `84.0 ms/tok` ✕ (cross-CPU PCIe) | `18.5 ms/tok` | `46.0 ms/tok` | 🏆 **`14.2 ms/tok` (`4.8 TB/s` HBM3e + `TP=4` NVLink)** |
| **Legacy `<6kW` Rack Compatibility** | `5.2 kW` | `10.2 kW` ✕ ($140K retrofit) | `3.8 kW` | 🏆 **`3.6 kW` (`4x L40S`) or `10.2 kW` (`8x H200`)** |
| **Single-GPU Fault Survivability** | `0%` ✕ (`TP=8` kills whole box) | `0%` ✕ (if configured `TP=8`) | `0%` | 🏆 **`50%+` capacity survives (dual `TP=4` replicas)** |
| **3-Year Cost / Concurrent User** | `$11,400 / user` ✕ | `$15,200 / user` ✕ (@30 users) | `$6,900 / user` | **`$2,350 / user` 🏆 (-79.4% vs Tier 1)** |

---

## ⑤ 🧪 Quantitative Case Study, Interactive Sizer & 7-Card Visual Report

- **Interactive HTML Sizer**: 👉 [Open On-Premises Private LLM Appliance & GPU Server Sizer (L40S vs. H100 vs. H200)](../reports/enterprise-private-llm-appliance-gpu-server-sizing-h200-l40s-sizer.html)
- **7-Card 4:5 Visual Benchmark Report**: 👉 [Open 7-Card Visual Benchmark Deck](../reports/enterprise-private-llm-appliance-gpu-server-sizing-h200-l40s-selection-cards.html)
- **Deterministic Verification**:
  ```bash
  python experiments/enterprise_private_llm_appliance_gpu_server_sizing_h200_l40s_selection/run.py --check
  python -m unittest tests/test_enterprise_private_llm_appliance_gpu_server_sizing_h200_l40s_selection.py
  ```

---

## ⑥ 🛠️ Production Readiness & Architecture Review Checklist

- [ ] **Verify Rack Power (`kW`) and Cooling Before Ordering `SXM` Servers**: Deploy `4U 4x L40S (3.6 kW)` in legacy `< 6 kW` air-cooled rooms; deploy `8x H200 SXM (10.2 kW)` only in `>= 10–12 kW` racks.
- [ ] **Cap Tensor Parallelism at `TP <= 4` on PCIe Servers Without NVSwitch**: Never span `TP=8` across dual-CPU socket UPI interconnects.
- [ ] **Enable `--kv-cache-dtype fp8` and `--enable-prefix-caching` by Default**: Cut per-user 32K KV Cache footprint from `10.24 GB` down to `~2.5 GB` effective VRAM.
- [ ] **Run Dual `TP=4` Replicas Instead of a Monolithic `TP=8` Instance on 8-GPU Nodes**: Ensure a single GPU ECC/Xid hardware failure degrades capacity by 50% rather than causing a 100% outage.
- [ ] **Use Hardware `NVIDIA MIG` on GPU 7 to Fence `TEI Embedding/Reranker` and `PDF OCR`**: Prevent document ingestion memory spikes from crashing active LLM chat sessions.
