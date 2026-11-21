# LLM Inference Acceleration Panorama & Parallelism Slicing (`TP` / `PP` / `Attention DP` / `MoE EP [DeepEP]` / `Speculative Decoding MTP & EAGLE-3` / `Prefix Caching` / `FP8 & NVFP4`) Selection Guide

> **Target Audience**: Cloud & AI Infrastructure Solution Architects, LLM Inference Optimization Engineers, and GPU/TPU Cluster Architects designing production serving topologies for `70B / 405B Dense` and `DeepSeek-V3/R1 671B Sparse MoE` models.
> **Core Question**: Across the 7 core inference acceleration levers—**Tensor Parallelism (`TP`), Pipeline Parallelism (`PP`), Data Parallelism (`Attention DP`), Expert Parallelism (`MoE EP + DeepEP`), Speculative Decoding (`MTP / EAGLE-3`), Prefix Caching (`RadixAttention / APC`), and Quantization (`FP8 / NVFP4`)**—which specific metric does each lever optimize (**`TTFT`**, **`TPOT`**, or **`System TPS`**), and what are the non-negotiable intra-node `NVLink` vs. inter-node `RDMA` slicing boundaries? Why does running `DeepSeek-671B` with dense-style `Cross-Node TP=16 in BF16` on a 64-GPU `H200` cluster (Tier 1) stall at **`46.5ms` P99 TPOT and `24,800 tok/s` (`$1.28/1M tokens`)**, whereas combining **`FP8 + Attention DP=64 × Expert EP=64 (DeepEP IBGDA + EPLB) + Native MTP Speculative Decoding + Prefix Caching` (Tier 4)** cuts P99 `TPOT` to **`11.6ms`**, multiplies cluster throughput **`5.18x` (`128,400 tok/s`)**, and slashes token cost by **`80.5%` (`$1.28 -> $0.25/1M`)**?

![LLM Inference Acceleration Panorama: TP / PP / DP / EP / MTP / Prefix / FP8 Summary](../../assets/svg/llm-inference-acceleration-tp-pp-dp-ep-speculative-prefix-fp8-selection-faq-summary.svg)

---

## 1. 📌 Executive Summary (3-Minute Verdict)

### 1.1 Master 7-Lever Reference Table: What Optimizes `TTFT` vs. `TPOT` vs. `System TPS`?

| Acceleration Lever | Underlying Mechanism | Improves `TTFT`? | Improves `TPOT`? | Improves `System TPS`? | Hardware & Fabric Boundary Rule |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. `TP` (Tensor Parallel)** | Splits weight matrices column/row-wise across $N$ GPUs, pooling $N\times$ FLOPs & HBM BW | **✅ Strong** (parallel Prefill) | **✅ Strong** (splits weight read) | ⚠️ Diminishing returns above `TP=4/8` | **⛔ Never cross inter-node RDMA (`TP <= 8` on HGX, `<=72` on NVL72)! Runs `2L` sync `All-Reduces`/step** |
| **2. `PP` (Pipeline Parallel)** | Splits layer ranges sequentially across nodes (`L1..40` on Node 1, `L41..80` on Node 2) | ❌ No | ❌ No | **✅ Strong** (fits 405B+ across nodes without cross-node TP) | **Built for inter-node `RDMA` boundaries** (only 1 activation `Send/Recv` at stage boundary) |
| **3. `Attention DP` (Data Parallel)** | Each rank processes an independent request batch with zero cross-GPU Attention traffic | ➖ Neutral | ✅ Eliminates Attention `TP` sync tax | **✅ Scales `System TPS` linearly** (0 KV duplication) | Crosses any node boundary; pairs with `MoE EP` on `DeepSeek-671B` |
| **4. `MoE EP` (`DeepEP` + `EPLB`)** | Places 4–8 full experts per GPU across 32–64 GPUs; routes tokens via `Dispatch` & `Combine` | ✅ Higher expert GEMM efficiency | **✅ Strong on MoE Decode** | **✅ Lifts MoE TPS `2.65x+`** | Requires `2L` `All-to-All` (`Dispatch + Combine`) over **`NVLink + GPUDirect RDMA (IBGDA)`** |
| **5. `Speculative Decoding` (`MTP / EAGLE-3`)** | Drafts 3–5 candidate tokens verified in a single target forward pass | ❌ No | **✅ Lossless `1.8x–2.5x` faster TPOT** | ✅ Boosts TPS at low/medium batch (`B <= 32`) | Raises Decode arithmetic intensity by `3–4x` using spare TensorCore FLOPs |
| **6. `Prefix Caching` (`RadixAttention / APC`)** | Reuses KV blocks for shared system prompts, RAG docs & multi-turn chat history | **✅ Cuts `TTFT` `3x–5x`** | ➖ Neutral | **✅ Saves Prefill FLOPs** | Pair with prefix-affine ingress routing |
| **7. Quantization (`FP8 / NVFP4 / AWQ`)** | Halves/quarters weight & KV HBM traffic and doubles TensorCore TFLOPS | **✅ `2x` Prefill speed** | **✅ `~1.8x` Decode speed** | **✅ `2x+` KV batch capacity** | Default to **Block-Scaled `FP8`** on Hopper/Ada and **`NVFP4`** on Blackwell |

### 1.2 4-Tier Benchmark on a 64-GPU `NVIDIA H200` Cluster (`DeepSeek-V3/R1 671B`)

| Parallelism & Acceleration Tier | Best-Fit Model Class | `P99 TTFT` | `P99 Decode TPOT` | 64-GPU Cluster `System TPS` | Cost / 1M Output Tokens (`$/1M`) | Architect Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Tier 1: Cross-Node `TP=16` (`BF16`) on `671B MoE` (Anti-Pattern)** | ❌ **Never run cross-node TP or pure TP on large MoE** | `890 ms` | **`46.5 ms` (SLA breach!)** | `24,800 tok/s` | `$1.28 / 1M` | ❌ **Reject**: 122 cross-node `All-Reduces`/token & 1-token-per-expert starvation |
| **Tier 2: Intra-Node `TP=8` × Inter-Node `PP=2` (`FP8`)** | **`405B Dense` on 2x 80GB H100 Nodes** | `510 ms` | `28.4 ms` | `52,400 tok/s` | `$0.61 / 1M` | ✅ **405B Dense Standard**: Avoids cross-node TP, but suboptimal for 256-expert MoE |
| **Tier 3: `FP8` + `Attention DP=64 × Expert EP=64` (No `EPLB` or `MTP`)** | Baseline multi-node MoE serving | `340 ms` | `21.8 ms` | `78,500 tok/s` | `$0.41 / 1M` | ⚠️ **Skew-Bottlenecked**: `2.45x` Zipf hot-expert imbalance stalls `All-to-All` barriers |
| **Tier 4: `FP8` + `Attention DP × Expert EP=64` (`DeepEP + EPLB`) + `MTP` + Prefix Cache (Recommended)** | **`DeepSeek-V3/R1 671B` Production Gold Standard** | **`145 ms`** | **`11.6 ms` (`4.0x` faster)** | **`128,400 tok/s` (`5.18x` Tier 1!)** | **`$0.25 / 1M` (`-80.5%`)** | 🏆 **Production Gold Standard**: Zero Attention sync, balanced experts, multi-token verify |

---

## 2. 🎯 Customer & Architect Critical Q&A

### Q1: Why does `TP=16` starve a `256-Expert MoE` model (`DeepSeek-671B`), and how does `Attention DP=64 × Expert EP=64` solve it?
* **Per-Expert Batch Starvation Under Pure `TP` vs. Batch Consolidation Under `Attention DP × Expert EP`**:
  * Under `TP=16`, every GPU holds a `1/16` slice of all 256 experts. At batch size `B=32` with Top-8 routing, each expert receives an average of $32 \times 8 / 256 = \mathbf{1\text{ token}}$—forcing TensorCores to execute degenerate `1-row` matrix-vector multiplies (`AI = 1 FLOP/Byte`) followed by cross-node `All-Reduce` barriers.
  * Under **`Attention DP=64 × Expert EP=64`**, all 64 GPUs run independent batches (`64 × 32 = 2,048` concurrent tokens across the cluster) with **zero cross-GPU communication in Attention**, and each GPU hosts **4 complete experts (`256 / 64 = 4`)**. Before the MoE FFN, **`DeepEP Dispatch (`All-to-All`)`** routes tokens from all 64 GPUs to the exact GPU holding their target expert—consolidating **$2,048 \times 8 / 256 = \mathbf{64\text{ tokens per expert}}$** (`64x` higher arithmetic intensity!) before **`DeepEP Combine (`All-to-All`)`** returns the outputs.

### Q2: How do `DeepEP` (`NVLink -> IBGDA`), Two-Microbatch Overlap, and `EPLB` work together?
1. **`DeepEP` Dual Kernels**: Uses high-throughput two-stage **`NVLink -> RDMA` forwarding** for Prefill, and low-latency **pure-`IBGDA` (GPU-initiated RDMA with `0-SM` background hooks)** for Decode.
2. **Two-Microbatch Ping-Pong Overlap**: Splits the active batch into `Microbatch A` and `Microbatch B` so `Microbatch A`'s `DeepEP All-to-All Dispatch/Combine` executes over RDMA while TensorCores compute `Microbatch B`'s `MLA Attention` (`85%+` communication hidden).
3. **`EPLB` (Expert Parallel Load Balancer)**: Dynamically replicates the top 16 hottest experts across nodes, flattening Zipf expert skew from `2.45x` to **`1.12x`** (`+53%` throughput).

### Q3: [Architect Cheat Sheet] What are the canonical slicing recipes for `70B Dense`, `405B Dense`, and `671B MoE`?

| Model Family | Hardware Footprint | Prefill Slicing Recipe | Decode Slicing Recipe | Key Accelerators & Target Metrics |
| :--- | :--- | :--- | :--- | :--- |
| **`70B Dense`** (`Llama-3.3-70B`) | `1x 8-GPU H200 / H100` | `FP8 + Intra-Node TP=4 (DP=2)` | `FP8 + Intra-Node TP=4 (DP=2)` (or `TP=8` for min latency) | `EAGLE-3` Speculative Decoding + `Prefix Cache` (`9.4ms` TPOT) |
| **`405B Dense`** (`Llama-3.1-405B`) | `1x 8-GPU H200 (1,128GB)` or `2x 8-GPU H100 (80GB)` | 1x H200: `FP8 + TP=8`<br/>2x H100: `FP8 + TP=8 × PP=2` | 1x H200: `FP8 + TP=8`<br/>2x H100: `FP8 + TP=8 × PP=2` | **Never use cross-node `TP=16` on 2x H100**; use `TP=8 × PP=2` |
| **`671B Sparse MoE`** (`DeepSeek-V3/R1`) | Compact: `2x 8-GPU H200`<br/>Scale: `8–18x 8-GPU H200` | `32 GPUs`: `FP8 + Attn DP=8 × TP=4 + EP=32` (`DeepEP` High-TP) | `64 GPUs`: `FP8 + Attn DP=64 × TP=1 + EP=64` (`DeepEP IBGDA`) | **`EPLB` + Native `MTP` + Two-Microbatch Overlap** (`11.6ms` TPOT, `128.4K tok/s`) |

---

## 3. 🧭 Visual Diagrams (Mermaid Slicing Comparison + Decision Tree + Sequence Diagram)

### 3.1 `TP × PP` (Dense 405B) vs. `Attention DP × Expert EP` (Sparse MoE 671B)

```mermaid
flowchart TB
    subgraph Dense["🧱 Dense 405B on 2x 80GB H100 Nodes: Intra-Node TP=8 (NVLink) × Inter-Node PP=2 (RDMA)"]
        direction LR
        N1["Node #1 (8x GPU NVLink 900 GB/s)\nLayers 1..63: Intra-Node TP=8\n(2x Fast NVLink All-Reduces / Layer)"] ==>|"Inter-Node RDMA Carries Only 1\nActivation Send/Recv at Layer 63"| N2["Node #2 (8x GPU NVLink 900 GB/s)\nLayers 64..126: Intra-Node TP=8\n(Zero Cross-Node TP Collectives!)"]
    end

    subgraph MoE["🚀 Sparse 671B MoE Decode on 64 GPUs: Attention DP=64 × Expert EP=64 (DeepEP + EPLB)"]
        direction LR
        ATTN["64 GPUs Run Independent MLA Attention\n(Attn DP=64, TP=1: Zero Comm & Zero KV Dup!)"] ==>|"DeepEP IBGDA Dispatch\n(All-to-All Top-8 Token Routing)"| EXP["64 GPUs Host 4 Full Experts Each (EP=64 + EPLB)\n64x Larger Batch Saturates FP8 TensorCores!"]
        EXP ==>|"DeepEP IBGDA Combine\n(All-to-All Return to Origin DP Rank)"| ATTN2["Next Layer MLA Attention\n(2-Microbatch Ping-Pong Hides 85% Comm!)"]
    end
```

### 3.2 Inference Parallelism & Acceleration Selection Decision Tree

```mermaid
flowchart TD
    A["Enterprise LLM Inference Slicing Design"] --> B{"Step 1: Dense Transformer or Sparse MoE?"}
    B -- "Dense Model (8B / 70B / 405B)" --> C{"Does FP8 Model + KV Pool fit inside 1 Node (8x NVLink GPUs)?"}
    C -- "Yes (70B, or 405B FP8 on 1x 8-GPU H200/B200)" --> D["✅ Pin Intra-Node TP=4 or TP=8 + Replicated DP\nEnable EAGLE-3 Speculative Decoding + Prefix Caching"]
    C -- "No (405B on 2x 80GB H100 Nodes)" --> E["✅ Intra-Node TP=8 (NVLink) × Inter-Node PP=2 (RDMA)\n(Strictly forbid Cross-Node TP=16!)"]

    B -- "Sparse MoE (DeepSeek-V3/R1 671B)" --> F{"Deployment Scale?"}
    F -- "Compact 2-Node (16x H200)" --> G["FP8 + Intra-Node TP=8 × Inter-Node PP=2 (or EP=16)"]
    F -- "High-Throughput Fleet (>=64 GPUs)" --> H["🏆 Disaggregated PD + Attention DP × Expert EP (Tier 4)\n• Prefill: Attn DP=8 × TP=4 + EP=32\n• Decode: Attn DP=64 × TP=1 + EP=64 + DeepEP IBGDA + EPLB + MTP"]
```

### 3.3 End-to-End Two-Microbatch Overlap, `DeepEP IBGDA` & `MTP` Verification Sequence

```mermaid
sequenceDiagram
    autonumber
    participant TC as GPU TensorCores (Compute Stream)
    participant RDMA as DeepEP IBGDA (0-SM Comm Stream)
    participant EPLB as EPLB Hot-Expert Router

    TC->>TC: Computes Layer $L$ `MLA Attention` for Microbatch A (`Attn DP=64`, zero comm)
    TC->>EPLB: Resolves Top-8 expert destinations (routes hot experts across replicated `EPLB` slots)
    par 100% Overlapped Compute & All-to-All Dispatch
        RDMA->>RDMA: [0-SM Hook] Runs `DeepEP IBGDA Dispatch` (`All-to-All`) for Microbatch A
    and
        TC->>TC: [TensorCore] Simultaneously computes Layer $L$ `MLA Attention` for Microbatch B!
    end
    par Overlapped Expert GEMM & Next Dispatch
        TC->>TC: [TensorCore] Executes `Grouped FP8 GEMM` on local 4 experts for Microbatch A
    and
        RDMA->>RDMA: [0-SM Hook] Simultaneously runs `DeepEP IBGDA Dispatch` for Microbatch B!
    end
    RDMA->>TC: Completes `DeepEP Combine` + `MTP` verifies 3 draft tokens in 1 pass (`P99 TPOT = 11.6ms`!)
```

---

## 4. 📊 4-Tier Architecture Decision Matrix

| Evaluation Dimension | Tier 1: Cross-Node `TP=16` (`BF16`) on `671B MoE` | Tier 2: Intra-Node `TP=8` × Inter-Node `PP=2` (`FP8`) | Tier 3: `FP8` + `Attn DP=64 × EP=64` (No `EPLB/MTP`) | Tier 4: `FP8` + `Attn DP × EP=64` (`DeepEP+EPLB`) + `MTP` + Prefix Cache (Recommended) |
| :--- | :--- | :--- | :--- | :--- |
| **`P99 Decode TPOT`** | **`46.5 ms` (122 cross-node AllReduces)** | `28.4 ms` | `21.8 ms` | **`11.6 ms` (`4.0x` faster than Tier 1!)** |
| **`P99 Prefill TTFT`** | `890.0 ms` | `510.0 ms` | `340.0 ms` | **`145.0 ms` (`RadixAttention` Prefix Cache + `FP8`)** |
| **64-GPU Cluster `System TPS`** | `24,800 tok/s` (`387/GPU`) | `52,400 tok/s` (`818/GPU`) | `78,500 tok/s` (`1,226/GPU`) | **`128,400 tok/s` (`2,006 tok/s/GPU`, `5.18x` Tier 1!)** |
| **Expert GEMM Batch & Load Balance** | ❌ 1 token/expert starvation | ❌ Replicates all 256 experts | ⚠️ `2.45x` Zipf hot-expert straggler | **✅ 64+ tokens/expert + `EPLB` flattens skew to `1.12x`** |
| **Cost / 1M Output Tokens (`$/1M`)** | `$1.28 / 1M` | `$0.61 / 1M` | `$0.41 / 1M` | **`$0.25 / 1M` (`-80.5%` savings)** |

---

## 5. 🧪 Quantitative Case Study, Interactive H5 Sizer & 7-Card Report

### 5.1 Baseline Workload (`64`-GPU `NVIDIA H200` Cluster Serving `DeepSeek-671B` & `Llama-3.1-405B`)
* **Workload Profile**: `64`-GPU `H200` cluster (`8` nodes connected over `400G RoCEv2/IB`) serving `DeepSeek-V3/R1 671B` and `Llama-3.1-405B` under a `P99 TPOT <= 20ms` SLA.
* **Reproducible Model & Dataset**: See [`experiments/llm_inference_acceleration_tp_pp_dp_ep_speculative_prefix_fp8_selection/run.py`](../../experiments/llm_inference_acceleration_tp_pp_dp_ep_speculative_prefix_fp8_selection/run.py) and [`results.json`](../../experiments/llm_inference_acceleration_tp_pp_dp_ep_speculative_prefix_fp8_selection/results.json) (`provenance: synthetic + anonymized_reported`, `independently_reproduced=false`).
* **Interactive H5 Sizer**: 📊 [Open LLM Inference Parallelism (`TP/PP/DP/EP`) & Acceleration Sizer (`llm-inference-acceleration-tp-pp-dp-ep-speculative-prefix-fp8-sizer.html`)](../reports/llm-inference-acceleration-tp-pp-dp-ep-speculative-prefix-fp8-sizer.html)
* **7-Card 4:5 Benchmark Deck**: 🃏 [Open 7-Card Visual Report (`llm-inference-acceleration-tp-pp-dp-ep-speculative-prefix-fp8-selection-cards.html`)](../reports/llm-inference-acceleration-tp-pp-dp-ep-speculative-prefix-fp8-selection-cards.html)

```chart
{
  "data": {
    "values": [
      { "tier": "Tier 1: Cross-Node TP=16 BF16", "p99_tpot_ms": 46.5, "cluster_tps_k": 24.8, "cost_per_1m_usd": 1.28 },
      { "tier": "Tier 2: Intra-Node TP=8 x PP=2 FP8", "p99_tpot_ms": 28.4, "cluster_tps_k": 52.4, "cost_per_1m_usd": 0.61 },
      { "tier": "Tier 3: FP8 Attn DP=64 x EP=64 Raw", "p99_tpot_ms": 21.8, "cluster_tps_k": 78.5, "cost_per_1m_usd": 0.41 },
      { "tier": "Tier 4: FP8 DPxEP + DeepEP/EPLB + MTP", "p99_tpot_ms": 11.6, "cluster_tps_k": 128.4, "cost_per_1m_usd": 0.25 }
    ]
  },
  "semantic_types": {
    "tier": "Category",
    "p99_tpot_ms": "Quantity",
    "cluster_tps_k": "Quantity",
    "cost_per_1m_usd": "Price"
  },
  "chart_spec": {
    "chartType": "Bar Chart",
    "title": "Tier 4 (FP8 + Attention DP x Expert EP + DeepEP/EPLB + MTP) Lifts Cluster TPS 5.18x (128.4K tok/s)",
    "subtitle": "64-GPU H200 cluster output throughput (Thousands of tokens/sec) on 671B Sparse MoE",
    "encodings": {
      "x": "tier",
      "y": "cluster_tps_k",
      "color": "tier"
    }
  }
}
```

---

## 6. 🛠️ Architecture Review & Production Readiness Checklist

- [ ] **Zero Cross-Node `TP` Enforcement**: Is `TP` strictly bounded to intra-node `NVLink` (`TP <= 8` on HGX), using `PP` (for multi-node Dense) or `Attention DP × Expert EP` (for multi-node MoE) across node boundaries?
- [ ] **`Attention DP + Expert EP (DeepEP)` on Sparse MoE**: Are multi-node MoE deployments configured with `Attention DP` (`TP=1` in Decode) and `DeepEP IBGDA` `Dispatch/Combine`?
- [ ] **`EPLB` Dynamic Hot-Expert Replication & Two-Microbatch Overlap**: Is `EPLB` enabled alongside two-microbatch ping-pong overlap to hide `>=85%` of `All-to-All` latency and flatten Zipf expert skew?
- [ ] **Lossless Speculative Decoding (`MTP / EAGLE-3`)**: Are native `MTP` or `EAGLE-3` draft heads enabled on latency-sensitive Decode pools (`1.8x–2.5x` lower TPOT)?
- [ ] **Block-Scaled `FP8` (Weights + Activations + `KV Cache`)**: Is hardware `FP8` enabled across weights, GEMMs, and `KV Cache` (or `NVFP4` on Blackwell)?
