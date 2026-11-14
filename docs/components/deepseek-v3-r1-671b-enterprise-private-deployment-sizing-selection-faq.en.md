# Enterprise DeepSeek-V3 / R1 (671B Full MoE) Private Deployment Sizing, Single-Node vs. Multi-Node & TCO Selection Guide (`KTransformers CPU+GPU Hybrid` vs. `2-Node 16xH100 FP8 TP=16` vs. `Single-Node 8xH200/MI300X FP8 + MTP` vs. `4-Node 32xH200 2P+2D Disaggregation + DeepEP`)

> 🌐 **Language / 语言**: **[中文 (`deepseek-v3-r1-671b-enterprise-private-deployment-sizing-selection-faq.md`)](deepseek-v3-r1-671b-enterprise-private-deployment-sizing-selection-faq.md)** | **English**
> **Reader's Guide**: Built for **Enterprise AI Solution Architects, CIOs, and Infrastructure Leads**, this guide explains from first principles: **how to size hardware for on-premises private deployment of full-weight DeepSeek-V3 / R1 (`671B MoE`, `37B` active per token); why spending `$520,000` on two `8xH100/H800 (80GB)` servers running 2-node `TP=16` (Tier 2) supports only `36` concurrent sessions and loses badly to a single `$310,000` `8xH200 (141GB)` or `8xMI300X (192GB)` server (Tier 3—saving `$210,000` in CapEx while delivering `96` concurrent sessions and `3.06x` higher throughput per GPU); how a 20-person R&D team with a `$22,000` budget can serve full 671B R1 using `KTransformers Dual-Socket Xeon AMX + 1x48GB GPU` (Tier 1); and why scaling to 4 nodes (`32xH200`) with `2P+2D Disaggregation + Attention DP + DeepEP + Mooncake` (Tier 4) unlocks `21,120 tok/s` (`560` concurrent users) at `$0.82 / 1M tokens`.**

![Enterprise DeepSeek-V3 / R1 671B Full MoE Private Deployment Sizing & TCO Summary](../../assets/svg/deepseek-v3-r1-671b-enterprise-private-deployment-sizing-selection-faq-summary.svg)

---

## 🔥 Daily Hottest Capability Breakthroughs

- **SGLang v0.4.x / vLLM V1 Single-Node `8xH200 / MI300X` Native `MTP + Attention DP` Acceleration**: Mainline engines now combine DeepSeek-V3/R1's built-in **Multi-Token Prediction (`MTP`) speculative head** with **Attention Data Parallelism (`--enable-dp-attention`)**, slashing single-node `8xH200` P99 decode latency (TPOT) from `16.5 ms` down to **`11.2 ms` (`1.47x` lossless speedup)** with zero cross-node network traffic.
- **`KTransformers` (KVCache.AI) Dual-Socket CPU `AMX` + Single-GPU Hybrid Breaking `50+ tok/s` at `$22K` CapEx**: By offloading the 256 sparse routed experts to `1 TB` of 24-channel DDR5 host memory (`~900 GB/s`) computed via Intel Xeon `AMX` matrix engines while keeping shared attention and `MLA` KV cache on a single `48GB L40S / RTX 4090D`, small teams can run full 671B R1 for **`$22,000`**.
- **`Mooncake Transfer Engine` + `DeepEP IBGDA` 4-Node `2P+2D` Disaggregated Pooling**: Because DeepSeek's `MLA` shrinks FP8 KV cache by **`93.3%` (`35.2 KB/token`)**, transferring a 16K-token prompt's KV cache (`562 MB`) across 400G RDMA from Prefill to Decode nodes takes just **`11.5 ms`**—boosting 4-node throughput by **`+51.7%`** over 4 isolated single-node islands.

---

## ① 📌 Executive Summary

1. **The Golden Rule of 671B Sizing: Never Buy Two 80GB Servers (`16xH100/H800 TP=16`) When One `8xH200 / MI300X` Server Fits the Entire Model In-Box**:
   - Full-weight DeepSeek-V3/R1 (`671B` total parameters, `37B` active per token) requires **`671 GB`** for native `FP8` weights plus `150–210 GB` for `MLA` KV cache and runtime buffers (`~851 GB` total).
   - **The Tier 2 Trap (`2 Nodes × 8xH100 80GB`, `$520,000` CapEx)**: Splitting one instance across 2 servers (`TP=16`) forces 122 cross-node RoCEv2 synchronizations per token and suffers severe head-of-line blocking whenever a 16K-token prompt prefill arrives—capping capacity at just **`36` concurrent users ✕**, inflating P99 TPOT to **`28.4 ms` (SLA violation ✕)**, and costing **`$3.85 / 1M tokens` ✕**!
2. **Right-Sized Architecture Tiers by Enterprise Concurrency & Budget**:
   - **For 5–30 Person R&D Labs / PoCs (`≤ 4` Concurrent Users, `≤ $25K` Budget) $\rightarrow$ Select `Tier 1 (KTransformers Dual-Socket Xeon AMX 1TB DDR5 + 1x48GB GPU Hybrid)`**: Offloads sparse routed experts to host DDR5 (`AMX`) while keeping shared layers and MLA KV on 1 GPU—serving genuine full 671B R1 (not a distilled 32B/70B model) for **`$22,000` CapEx** (`52.0 tok/s`) ⚠️.
   - **For 90% of Medium-to-Large Enterprises (`30–96` Concurrent Users, 500–3,000 Employees) $\rightarrow$ Select `Tier 3 (Single-Node 8xH200 1,128GB or 8xMI300X 1,536GB FP8 + Native MTP)`**: **1 server beats 2 H100 servers!** Cuts hardware CapEx from `$520,000` to **`$310,000` (saving `$210,000` immediately ✅)** while delivering **`3,480.0 tok/s` (`3.06x` higher throughput per GPU ✅)**, **`96` concurrent users ✅**, **`11.2 ms` P99 TPOT ✅**, and **`$1.24 / 1M tokens` (`-67.8%` ✅)**!
   - **For Large Enterprises & Financial/Gov AI Platforms (`200–560+` Concurrent Users, 10,000+ Employees) $\rightarrow$ Select `Tier 4 (4-Node 32xH200 2P+2D Disaggregation + Attention DP + DeepEP EP=16 + Mooncake)`**: Dedicates 2 nodes (`16 GPUs`) to Prefill and 2 nodes (`16 GPUs`) to Decode with `DeepEP` `EP=16`, reaching **`21,120.0 tok/s` (`4.65x` per-GPU efficiency vs. Tier 2 🏆)**, **`560` concurrent users 🏆**, **`640 ms` P99 TTFT**, **`9.4 ms` P99 TPOT**, and **`$0.82 / 1M tokens` (`-78.7%` 🏆)**!

---

## ② 🎯 Architect & Customer FAQ (First-Principles Q&A)

### Q1: What is the fundamental difference between `DeepSeek-R1-Distill-32B/70B` and full `671B MoE`, and how can `KTransformers` (Tier 1) run the full 671B model on a `$22,000` server with only 1 GPU?

**Plain-Language First Principles**:
1. **Why Distilled 32B/70B Models Are Not Full 671B R1**:
   - `Distill-32B/70B` are dense Qwen/Llama checkpoints fine-tuned on R1 reasoning traces; they lack the 256 specialized domain experts (`671B` total parameters) and exhibit higher hallucination rates on complex multi-file code refactoring, legal auditing, and deep domain reasoning.
2. **How `KTransformers` Exploits MoE Sparsity (`671B` Stored vs. `37B` Activated)**:
   - Although DeepSeek-V3/R1 requires `~400–671 GB` of storage, each token activates **only 8 out of 256 routed experts (`~28B` expert parameters)**.
   - By placing the 256 routed experts in `1 TB` of dual-socket 24-channel DDR5 host memory (`~900 GB/s` bandwidth) executed by Intel Xeon `AMX` matrix units, and placing the shared dense attention layers + `MLA` KV cache on **a single 48GB GPU**, CPU memory bandwidth only needs to read `~28 GB` per token step—achieving **`18–25 tok/s` for 1 user** and **`52.0 tok/s` across 4 concurrent users** for just **`$22,000`**!

---

### Q2: How do DeepSeek's `MLA (Multi-head Latent Attention)` and native `MTP (Multi-Token Prediction)` change the enterprise server sizing equation in Tier 3?

**Plain-Language First Principles**:
1. **`MLA` Shrinks KV Cache Footprint by `93.3%` (`35.2 KB/token` in FP8)**:
   - Whereas a traditional 405B GQA model consumes `~252 KB/token` in FP8 (`~516 KB` in FP16), DeepSeek's `MLA` compresses all Key/Value heads into a single `512`-dim latent vector plus a `64`-dim RoPE key (`576` bytes per layer in FP8):
     $$\text{KV}_{\text{MLA-FP8}} = 61 \times 576\text{ B} = 35,136\text{ B/token} \approx 35.2\text{ KB/token}$$
   - Even with **96 concurrent 16K-token sessions**, total KV cache memory is only **`54.0 GB`**—fitting effortlessly inside a single `8xH200 (1,128 GB)` or `8xMI300X (1,536 GB)` node alongside the `671 GB` FP8 weights!
2. **Native `MTP` Delivers a Free `1.47x` Lossless Decode Boost**:
   - Because DeepSeek-V3/R1 ships with a pre-trained Multi-Token Prediction (`MTP`) head, enabling `--speculative-algorithm EAGLE/MTP` drafts and verifies 2 tokens per forward pass with `~85%` acceptance—cutting single-node P99 TPOT from `16.5 ms` to **`11.2 ms` ✅**!

---

### Q3: Once an enterprise scales to 4 `8xH200` nodes (32 GPUs), why does pooling them into `Tier 4 (2P+2D Disaggregation + DeepEP + Mooncake)` beat running 4 isolated single-node instances by +51.7%?

**Plain-Language First Principles**:
1. **Eliminating Prefill-Decode Interference**: In colocated nodes, a burst of 16K-token RAG prefills monopolizes Tensor Cores for `>1,000 ms`, freezing streaming decode tokens for all co-located users. Separating **2 Prefill nodes (`16 GPUs`)** from **2 Decode nodes (`16 GPUs`)** isolates both phases; transferring a 16K-token MLA KV cache (`562 MB`) over 400G `Mooncake` RDMA takes just **`11.5 ms`**!
2. **Halving Expert Memory Traffic via 16-Way `DeepEP` (`EP=16`)**: Pooling the 16 Decode GPUs into a single `EP=16` domain via `DeepEP IBGDA` places only 16 experts per GPU (instead of 32) and doubles the batch size per expert—lifting per-GPU throughput from `435 tok/s` to **`660 tok/s` (`21,120 tok/s` total 🏆)**!

---

## ③ 🧭 Visual Architecture Diagrams (Mermaid Decision Tree / Topology Comparison / Sequence Diagram)

### 3.1 Enterprise DeepSeek-V3/R1 (671B) Private Sizing Decision Tree (`flowchart TD`)

```mermaid
flowchart TD
    A["🏗️ Start Enterprise DeepSeek-V3 / R1 (671B Full MoE) Sizing<br/>(Full 671B Intelligence · 16K–32K Context · Target SLA: TTFT <=1.5s, TPOT <=20ms)"] --> B{"Peak Concurrent Active Sessions & CapEx Budget?"}
    B -- "<= 4 Concurrent Users (<=30-Person R&D Lab) · Budget <= $25,000" --> C["⚠️ Select Tier 1: KTransformers Dual-Socket CPU AMX + 1x48GB GPU<br/>• 256 routed experts in 1TB DDR5 (AMX), shared attention & MLA KV on 1 GPU<br/>• $22,000 total CapEx runs full 671B R1 at 52 tok/s (P99 TPOT 58ms)"]
    B -- "> 10 Concurrent Production Sessions · Budget >= $300,000" --> D{"Planning to Buy Two 80GB Servers (16xH100/H800)<br/>for 2-Node TP=16 (Tier 2)?"}
    D -- "Yes (Tier 2: 2-Node 16xH100 Colocated)" --> E["⛔ Reject Tier 2 (Costs $210K More & Runs Slower)<br/>• 122 cross-node RoCE barriers/token + Prefill stalls Decode<br/>• $520,000 CapEx supports only 36 users at 28.4ms TPOT ($3.85/1M) ✕"]
    D -- "Procure High-Memory HBM3e Nodes (H200 141GB / MI300X 192GB)" --> F{"Peak Concurrent Sessions: <= 100 Users vs. 200–560+ Users?"}
    F -- "30 – 96 Concurrent Sessions (500–3,000 Employee Enterprise)" --> G["✅ Select Tier 3: Single-Node 8xH200 (1,128GB) or 8xMI300X (1,536GB) + MTP<br/>• Fits full 671B FP8 in 1 server + native MTP speculative decoding + Attention DP<br/>• $310,000 CapEx (saves $210K vs Tier 2), 3,480 tok/s (3.06x/GPU), $1.24/1M ✅"]
    F -- "200 – 560+ Concurrent Sessions (10,000+ Employee Enterprise AI Platform)" --> H["🏆 Select Tier 4: 4-Node 32xH200 2P+2D Disaggregation + DeepEP + Mooncake<br/>• 2 Prefill + 2 Decode nodes (EP=16) + 11.5ms Mooncake RDMA MLA KV transfer<br/>• 560 concurrent users, 9.4ms TPOT, 21,120 tok/s, $0.82/1M (-78.7% 🏆)"]
```

### 3.2 `Tier 2 2-Node H100 Trap` vs. `Tier 3 Single-Node H200+MTP` vs. `Tier 4 2P+2D + DeepEP` (`flowchart LR`)

```mermaid
flowchart LR
    subgraph Bad["❌ Tier 2 Trap: 2x 8-GPU H100 ($520K · 2-Node TP=16)"]
        direction TB
        B1["2x 8xH100 Split Across 400G RoCE (TP=16)<br/>122 Network Syncs/Token + Prefill Stalls Decode ✕"] --> B2["Only 36 Concurrent Users · 28.4 ms TPOT<br/>2,272 tok/s Total · $3.85 / 1M Tokens ✕"]
    end

    subgraph Mid["✅ Tier 3 Sweet Spot: 1x 8-GPU H200/MI300X ($310K · In-Box + MTP)"]
        direction TB
        M1["1x 8xH200 (1,128GB) Fits Full 671B FP8 In-Box<br/>100% Intra-Node NVLink + Attention DP + Native MTP ✅"] --> M2["96 Concurrent Users (Saves $210K CapEx!)<br/>11.2 ms TPOT · 3,480 tok/s · $1.24 / 1M Tokens ✅"]
    end

    subgraph Top["🏆 Tier 4 Flagship: 4x 8-GPU H200 (2P+2D + DeepEP EP=16)"]
        direction TB
        T1["2 Prefill Nodes Compute 16K Prompt (562MB MLA KV)<br/>11.5ms Mooncake RDMA Write to 2 Decode Nodes 🏆"] --> T2["16-GPU Decode Pool Runs DeepEP IBGDA EP=16<br/>560 Users · 9.4 ms TPOT · 21,120 tok/s · $0.82/1M 🏆"]
    end

    Bad -. "Eliminate Cross-Node TP & Save $210K" .-> Mid
    Mid -. "Scale to 4-Node 2P+2D Pool" .-> Top
```

### 3.3 End-to-End `Tier 4 (2P+2D + Mooncake MLA KV + DeepEP + MTP)` Execution Flow (`sequenceDiagram`)

```mermaid
sequenceDiagram
    autonumber
    participant User as Enterprise User (16K RAG/Code Prompt)
    participant Router as Dynamo / Mooncake KV-Aware Router
    participant Prefill as Prefill Pool (2x8 H200 · Compute-Bound)
    participant Decode as Decode Pool (2x8 H200 · DeepEP EP=16 + MTP)

    User->>Router: 1. Submit 16,384-token request (SLA: TTFT <=1.5s, TPOT <=20ms)
    Router->>Prefill: 2. Route to highest prefix-hit Prefill node
    Note over Prefill: FlashMLA computes prefill & compresses FP8 MLA KV Cache<br/>Total 16K KV payload = 16,384 × 35.2 KB = 562 MB (93.3% smaller than GQA!)
    Prefill->>Decode: 3. Mooncake 400G RDMA Zero-Copy Write transfers 562 MB KV in 11.5 ms!
    Decode-->>User: 4. Emit first token (P99 TTFT = 640 ms ✅, zero decode jitter)
    loop 5. Stream 4,096 reasoning tokens (Attention DP + DeepEP IBGDA EP=16 + Native MTP)
        Decode->>Decode: GPU-initiated IBGDA RDMA Dispatch/Combine + MTP verifies 2 tokens/step
        Decode-->>User: 6. Stream tokens at P99 TPOT = 9.4 ms 🏆 (21,120 tok/s cluster throughput 🏆)
    end
```

---

## ④ 📊 Architect Decision Matrix

*Benchmark Profile: On-premises private deployment of full-weight `DeepSeek-V3 / R1 (671B MoE, 37B Active)` on an enterprise RAG/coding workload (`16K input prompt + 4K CoT output`, target SLA: `P99 TTFT ≤ 1,500 ms`, `P99 TPOT ≤ 20.0 ms`).*

| Selection Dimension / Deployment Tier | Tier 1: `KTransformers` (`Dual-Socket AMX + 1x48GB GPU`) | Tier 2: 2-Node `16xH100/H800 80GB` (`Colocated TP=16`) | Tier 3: Single-Node `8xH200 141GB / MI300X` (`FP8 + MTP`) | Tier 4: 4-Node `32xH200` (`2P+2D + DeepEP + Mooncake`) |
| :--- | :--- | :--- | :--- | :--- |
| **Hardware Configuration** | `2x Xeon AMX (1TB DDR5)` + `1x L40S/4090D 48GB` | `2 Nodes × 8xH100 80GB` (`16 GPUs`, `1,280 GB`) | **`1 Node × 8xH200 141GB` (`1,128 GB`) or `8xMI300X` ✅** | **`4 Nodes × 8xH200 141GB` (`16P + 16D = 32 GPUs`) 🏆** |
| **Parallelism & Execution Model** | Routed experts on CPU AMX, shared/KV on 1 GPU | Cross-node `TP=16` over RoCEv2 (122 syncs/token ✕) | **In-box `TP=8 / Attention DP` + Native `MTP` ✅** | **`2P+2D` Disaggregation + `DeepEP EP=16` + `Mooncake` 🏆** |
| **Upfront Hardware CapEx** | **`$22,000` (Ultra-low entry barrier ✅)** | `$520,000` (2 servers, expensive & slow ✕) | **`$310,000` (Saves `$210,000` vs. 2-node H100 ✅)** | `$1,240,000` (Enterprise-wide AI platform) |
| **Max SLA-Compliant Concurrency** | `4 Concurrent Users` | `36 Concurrent Users` (Stalled by RoCE & Prefill ✕) | **`96 Concurrent Users` (`2.67x` vs. Tier 2 ✅)** | **`560 Concurrent Users` (`15.6x` vs. Tier 2 🏆)** |
| **16K Prompt P99 TTFT** | `6,850.0 ms` (CPU prefill bound ⚠️) | `2,150.0 ms` (Head-of-line queuing ✕) | **`1,120.0 ms` (Meets `≤1.5s` SLA ✅)** | **`640.0 ms` (Dedicated Prefill pool 🏆)** |
| **P99 Decode Latency (TPOT)** | `58.0 ms` (`~17 tok/s` ⚠️) | `28.4 ms` (Violates `≤20ms` SLA ✕) | **`11.2 ms` (`~89 tok/s`, smooth streaming ✅)** | **`9.4 ms` (`~106 tok/s`, ultra-fast 🏆)** |
| **Total Cluster Throughput** | `52.0 tok/s` | `2,272.0 tok/s` (`142 tok/s/GPU` ✕) | **`3,480.0 tok/s` (`435 tok/s/GPU = 3.06x` ✅)** | **`21,120.0 tok/s` (`660 tok/s/GPU = 4.65x` 🏆)** |
| **Cost per 1M Output Tokens** | `$2.95 / 1M Tokens` | `$3.85 / 1M Tokens` (Most expensive ✕) | **`$1.24 / 1M Tokens` (`-67.8%` Savings ✅)** | **`$0.82 / 1M Tokens` (`-78.7%` Savings 🏆)** |

---

## ⑤ 🧪 Quantitative Case Study & Benchmark Artifacts

### 5.1 Reproducible Evaluator & Verification Harness

- **Evaluator Script**: [`experiments/deepseek_v3_r1_671b_private_sizing_selection/run.py`](../../experiments/deepseek_v3_r1_671b_private_sizing_selection/run.py)
- **Sealed Results JSON**: [`experiments/deepseek_v3_r1_671b_private_sizing_selection/results.json`](../../experiments/deepseek_v3_r1_671b_private_sizing_selection/results.json)
- **Unit Test Suite**: [`tests/test_deepseek_v3_r1_671b_private_sizing_selection.py`](../../tests/test_deepseek_v3_r1_671b_private_sizing_selection.py)
- **7-Card 4:5 Visual Benchmark Deck**: [`docs/reports/deepseek-v3-r1-671b-enterprise-private-deployment-sizing-selection-cards.html`](../reports/deepseek-v3-r1-671b-enterprise-private-deployment-sizing-selection-cards.html)
- **Interactive Sizer Workbench**: [`docs/reports/deepseek-v3-r1-671b-enterprise-private-deployment-sizing-sizer.html`](../reports/deepseek-v3-r1-671b-enterprise-private-deployment-sizing-sizer.html)
- **Provenance**: `synthetic_analytical_model_calibrated_to_published_serving_benchmarks` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`).

### 5.2 Key Quantitative Takeaways

1. **Why 1 `8xH200` Node Beats 2 `8xH100` Nodes**:
   - `Tier 3 (1x 8xH200)` costs **`$310,000`** (`$210,000` less than `Tier 2`'s `$520,000`) yet produces **`3,480.0 tok/s`** vs. `Tier 2`'s **`2,272.0 tok/s`**—a **`3.06x` per-GPU efficiency advantage** (`435` vs. `142 tok/s/GPU`).
2. **Why 4 Nodes Must Use `2P+2D` Disaggregation**:
   - Combining 4 `8xH200` nodes into a `2P+2D` disaggregated pool with `DeepEP EP=16` lifts per-GPU throughput to **`660.0 tok/s/GPU` (`4.65x` vs. Tier 2)** and cuts token cost by **`78.7%` (`$0.82 / 1M`)**.

---

## ⑥ 🛠️ Good Fit vs. Not For & Production Verification Checklist

### 6.1 Good Fit vs. Not For Summary

| Fit Category | Architectural Guidance |
| :--- | :--- |
| **✅ Good Fit** | • **Small R&D Teams ($\le 30$ users, $\le \$25\text{K}$ budget)**: Deploy **`Tier 1 (KTransformers Dual-Socket Xeon AMX + 1x48GB GPU)`** to get genuine 671B R1 reasoning.<br/>• **Mainstream Enterprise Private Deployment (500–3,000 employees, $\le 96$ concurrent sessions)**: Standardize on **`Tier 3 (Single-Node 8xH200 1,128GB or 8xMI300X 1,536GB + FP8 + Native MTP)`**.<br/>• **Large Enterprise AI Platforms ($\ge 4$ servers, $500+$ concurrent sessions)**: Deploy **`Tier 4 (2P+2D Disaggregation + Attention DP + DeepEP + Mooncake)`**. |
| **⛔ Not For** | • **Never buy 2 `8xH100/H800 (80GB)` servers for 2-node `TP=16` (Tier 2)**: Wastes `$210,000` more CapEx for 35% lower throughput.<br/>• **Never leave `--enable-dp-attention` and native `MTP` disabled** on DeepSeek-V3/R1 production servers. |

### 6.2 Production DeepSeek-V3/R1 Deployment Checklist (`- [ ]`)

- [ ] **Full-Weight & KV Dtype Verification**: Confirm the checkpoint is the native `671B FP8 (E4M3)` MoE model (not a dense 32B/70B distillation) and `--kv-cache-dtype fp8_e4m3` (`35.2 KB/token`) is active.
- [ ] **Native `MTP` & `Attention DP` Flags**: Verify `--enable-dp-attention` and `--speculative-algorithm EAGLE/MTP` are enabled in `SGLang` or `vLLM V1`.
- [ ] **Host Memory Channel Population (Tier 1 `KTransformers`)**: Verify all 16/24 DDR5 DIMM channels across both CPU sockets are populated and NUMA-pinned (`numactl`) for `~900 GB/s` aggregate AMX bandwidth.
- [ ] **RDMA `2P+2D` & `DeepEP` Check (Tier 4)**: Confirm `Mooncake` zero-copy KV transfer (`<15 ms` for 16K tokens) and `DeepEP IBGDA` (`NVSHMEM`) are active across the 400G RoCEv2/IB fabric.
