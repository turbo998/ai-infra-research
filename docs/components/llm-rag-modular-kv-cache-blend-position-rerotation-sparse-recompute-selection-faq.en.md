# Enterprise RAG Non-Prefix Modular KV Reuse (`CacheBlend / LMCache`), `Pre-RoPE` Re-Rotation & `15%` Selective Sparse Recompute Selection Guide: Why Dynamic Document Ordering Causes `92%` Radix Prefix Misses (`2,850 ms TTFT`), Why Naive KV Concatenation Crashes Multi-Hop F1 by `-19.4 pt` & How `Pre-RoPE` Shift + `15%` High-Drift Token Recompute (`390 ms TTFT`, `7.31x` Faster, `78.1 F1`) Solves It

> **Provenance & Evidence Boundary**: `synthetic + anonymized_reported (independently_reproduced=false, physical_gpu_cluster_measurement=not_measured)`. All quantitative metrics in this guide are grounded in the sealed deterministic evaluator `experiments/llm_rag_modular_kv_cache_blend_position_rerotation_sparse_recompute_selection/run.py` and `results.json`.

![Non-Prefix RAG KV Reuse CacheBlend Visual Summary](../../assets/svg/llm-rag-modular-kv-cache-blend-position-rerotation-sparse-recompute-selection-faq-summary.svg)

---

## 0. Daily Capability Breakthrough

- **Today's Highlight Breakthrough**: **Position-Independent `Pre-RoPE` Modular KV Caching + Runtime Target-Offset `RoPE` Re-Rotation ($R(\Delta p)$) + Layer-1 Guided `15%` High-Drift Token Selective Cross-Attention Recomputation (`CacheBlend / LMCache`) for Dynamic Multi-Document RAG**.
- **Architect Value**: In enterprise RAG pipelines, each query retrieves $K=10$ document chunks (`~3,200` tokens each, `32,768` tokens total) and assembles them in query-specific relevance order `[System, Doc_1, Doc_2, ..., Doc_10, Query]`:
  1. **Standard Exact-Prefix Radix Cache (`vLLM APC / SGLang RadixAttention`, Tier 1)**: Because exact prefix caching hashes token sequences strictly from index `0`, changing the first retrieved chunk `Doc_1` invalidates the KV cache for `Doc_2 .. Doc_10` even if those 9 chunks were already computed hundreds of times—causing a **`92%` cache miss rate** and stalling `32K` TTFT at **`2,850.0 ms`**!
  2. **Naive Position-Independent KV Concatenation (`PromptCache`, Tier 2)**: Precomputing each document chunk's KV cache independently in isolation and concatenating them at runtime cuts TTFT to `210.0 ms`, but suffers two fatal flaws: ① `RoPE` positions are misaligned; ② `Doc_5` never attends to `Doc_1` (`0%` cross-document attention), crashing multi-hop reasoning accuracy from `78.4 F1` to **`59.0 F1` (`-19.4 pt` drop)**!
  By **rotating cached keys to their runtime token offsets ($R(p_{\text{new}} - p_{\text{old}})$) and selectively recomputing only the top `15%` high-drift tokens identified at Layer 1 across subsequent layers (`CacheBlend`, Tier 4)**, `32K` RAG TTFT drops **`7.31x` (`2,850.0 ms -> 390.0 ms`)** while saving **`84.6%` of prefill FLOPs** and restoring multi-hop accuracy to **`78.1 F1`**!

---

## ① 📌 Executive Summary

1. **Why `RoPE` Re-Rotation Alone (Tier 3, `66.2 F1`) Is Not Enough for Multi-Hop RAG**:
   - Applying relative rotation $R(p_{\text{new}} - p_{\text{old}})$ to precomputed keys fixes positional alignment, but each chunk's deeper-layer KV representations were still computed **without attending to preceding chunks**. Cross-document coreferences ("the company mentioned above") fail without cross-chunk attention.
2. **First-Principles Sparsity (`15%` Selective Cross-Attention Recompute in `CacheBlend`)**:
   - When chunk B is placed after chunk A, `~85%` of tokens in chunk B (function words, local clauses) have virtually identical KV vectors whether chunk A is present or not. Only **`~15%` of tokens (entity bridges, attention sinks, cross-references)** exhibit significant KV drift.
   - Comparing full vs. cached KV at Layer 1 identifies the top `15%` high-drift token indices $\mathcal{I}_{\text{imp}}$. In Layers $2..L$, gathering only those `15%` tokens into a compact dense matrix $H_{\text{imp}} \in \mathbb{R}^{0.15N \times D}$ for QKV/FFN projection and cross-chunk FlashAttention recovers **`99.6%` of full-prefill accuracy** at `15%` of the compute cost!
3. **Zero-Bubble I/O Overlap**:
   - The `15%` sparse compute time per layer matches the PCIe/NVLink transfer time needed to stream the next layer's `85%` cached KV blocks from Host DRAM (`LMCache`).

---

## ② 🎯 Architect Q&A (Q1 / Q2 / Q3)

### Q1: Can we re-rotate cached Keys if they were already stored *after* applying `RoPE` at position $p_{\text{old}}$?
- **Yes!** By the orthogonal group property of 2D rotary matrices, $R(p_{\text{new}} - p_{\text{old}}) \cdot (R(p_{\text{old}}) k) = R(p_{\text{new}}) k$. A single fused Triton kernel shifts any cached `Post-RoPE` key to its new runtime offset $p_{\text{new}}$ during H2D copy.

### Q2: How do we prevent `15%` sparse token recomputation from becoming memory-bound scattered reads?
- Use **Gather-Compact-Scatter**: gather the `4,915` selected token hidden states (`15%` of `32,768`) into a contiguous `[4915, D]` tensor so QKV and SwiGLU GEMMs run at full Tensor Core utilization, then scatter the updated `K_imp, V_imp` back into the Paged KV table before running FlashAttention.

### Q3: How should the selective recomputation ratio $r$ be tuned across workloads?
- Use `r = 10%` for independent factoid extraction and `r = 15%–20%` for multi-hop reasoning or cross-file repository code analysis.

---

## ③ 🧭 Visual Diagrams (Decision Tree + Comparison + Sequence)

### 3.1 Multi-Document RAG Modular KV Reuse Decision Tree

```mermaid
flowchart TD
    A["RAG Pipeline Assembles K=10 Retrieved Chunks (32K Tokens) in Dynamic Order"] --> B{"Are Chunk Positions & Prefix Order Always Static?"}
    B -- "Yes: Static System Prompt / Fixed Prefix" --> C["Use Standard Exact-Prefix Radix Cache (vLLM APC / SGLang)"]
    B -- "No: Dynamic Query-Dependent Chunk Ordering" --> D{"Modular Non-Prefix KV Reuse & Blending Strategy?"}
    D -- "Standard Exact-Prefix Cache" --> E["⛔ Tier 1: Changing Doc #1 invalidates Docs #2..10 (92% miss, 2,850 ms TTFT)"]
    D -- "Naive KV Concatenation (0% Cross-Chunk Recompute)" --> F["⛔ Tier 2/3: Zero cross-document attention -> Multi-Hop F1 drops -12.2 to -19.4 pt!"]
    D -- "RoPE Re-Rotation + CacheBlend 15% Selective Recompute" --> G["🏆 Tier 4: 390.0 ms TTFT (7.31x faster) + 78.1 F1 (near-lossless) + 84.6% FLOPs saved!"]
```

### 3.2 Exact-Prefix Miss (`2,850 ms`) / Naive Concat (`59.0 F1`) vs. `CacheBlend` `15%` (`390 ms`, `78.1 F1`)

```mermaid
flowchart LR
    subgraph Bad["⛔ Tier 1/2: 2,850 ms Prefix Miss or 59.0 F1 Naive KV Concat"]
      B1["Tier 1: Reordered Doc #1 Invalidates 92% of Prefix Cache (2,850 ms)"] --> B2["Tier 2: Naive KV Concat Loses Cross-Doc Attention (59.0 F1)"]
    end
    subgraph Good["🏆 Tier 4: Pre-RoPE Shift + 15% High-Drift Selective Recompute (390 ms, 78.1 F1)"]
      G1["Reuse 85% Cached KV with Runtime RoPE Shift R(p_new - p_old)"] --> G2["Gather & Recompute Top 15% High-Drift Tokens Across Chunks -> 78.1 F1!"]
    end
```

### 3.3 Pipelined Host KV Fetch & `15%` Selective Recompute Sequence

```mermaid
sequenceDiagram
    participant Store as Host DRAM / NVMe KV Store (LMCache)
    participant Copy as CUDA Copy Stream (Async H2D + RoPE Shift)
    participant Comp as CUDA Compute Stream (15% Sparse Recompute)

    Store->>Copy: 1. Fetch Layer 1 cached KV for Docs 1..10 & apply RoPE shift R(Δp)
    Copy->>Comp: 2. Layer 1 Probe: select Top-15% high-drift token indices I_imp
    par Pipelined Layers l = 2 .. 80
        Store->>Copy: 3a. Async H2D prefetch Layer l+1 cached KV (85% tokens) + RoPE shift
        Comp->>Comp: 3b. Gather H[I_imp] (15%) -> Dense QKV/FFN GEMM -> Scatter KV[I_imp] & FlashAttn
    end
    Note over Store,Comp: 4. Complete 32K RAG prefill in 390.0 ms (7.31x faster) at 78.1 F1!
```

---

## ④ 📊 4-Tier Architect Decision Matrix

| Cohort Tier | Modular KV Caching & Blending Strategy | `32K` RAG TTFT Latency | Multi-Hop RAG `F1` | Prefill FLOPs Saved | Best For |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Tier 1: Standard Exact-Prefix Cache** | Matches strictly from token index `0` | `2,850.0 ms` (92% Miss! ✕) | `78.4` (Baseline) | `8.0%` ✕ | ⛔ **Dynamic-Order Multi-Doc RAG** |
| **Tier 2: Naive Independent KV Concat** | Zero cross-chunk attn + unshifted RoPE | `210.0 ms` | `59.0` (`-19.4` Crash! ✕) | `94.5%` | ⛔ **Severe Multi-Hop Accuracy Loss** |
| **Tier 3: Dynamic `RoPE` Re-Rotation Only** | Shifts RoPE offset, zero cross-chunk attn | `245.0 ms` ✅ | `66.2` (`-12.2` Drop ⚠️) | `92.0%` | ⚠️ Independent Single-Chunk Lookup |
| **Tier 4: `RoPE` Shift + `15%` `CacheBlend` (Recommended)** | **Layer-1 guided `15%` sparse recompute** | **`390.0 ms` (`7.31x` Faster 🏆)** | **`78.1` (Near-Lossless 🏆)** | **`84.6%` 🏆** | 🏆 **Enterprise Multi-Hop RAG & Agents** |

---

## ⑤ 🧪 Quantitative Evaluator & 7-Card 4:5 Visual Report

- **Sealed Evaluator**: [`experiments/llm_rag_modular_kv_cache_blend_position_rerotation_sparse_recompute_selection/run.py`](../../experiments/llm_rag_modular_kv_cache_blend_position_rerotation_sparse_recompute_selection/run.py) · Data: [`results.json`](../../experiments/llm_rag_modular_kv_cache_blend_position_rerotation_sparse_recompute_selection/results.json)
- **Interactive Sizer**: [`docs/reports/llm-rag-modular-kv-cache-blend-position-rerotation-sparse-recompute-sizer.html`](../reports/llm-rag-modular-kv-cache-blend-position-rerotation-sparse-recompute-sizer.html)
- **7-Card 4:5 Visual Benchmark Deck**: [`docs/reports/llm-rag-modular-kv-cache-blend-position-rerotation-sparse-recompute-selection-cards.html`](../reports/llm-rag-modular-kv-cache-blend-position-rerotation-sparse-recompute-selection-cards.html)

```chart
{
  "data": {
    "values": [
      { "tier": "Tier 1: Exact-Prefix Radix Cache", "ttft_ms": 2850.0, "f1_score": 78.4 },
      { "tier": "Tier 2: Naive Independent KV Concat", "ttft_ms": 210.0, "f1_score": 59.0 },
      { "tier": "Tier 3: Pre-RoPE Re-Rotation Only", "ttft_ms": 245.0, "f1_score": 66.2 },
      { "tier": "Tier 4: CacheBlend 15% Sparse Recompute", "ttft_ms": 390.0, "f1_score": 78.1 }
    ]
  },
  "semantic_types": {
    "tier": "Category",
    "ttft_ms": "Quantity",
    "f1_score": "Score"
  },
  "chart_spec": {
    "chartType": "Bar Chart",
    "title": "CacheBlend (RoPE Re-Rotation + 15% Sparse Recompute) Cuts 32K RAG TTFT 7.31x Without Losing Multi-Hop F1",
    "subtitle": "32K-context RAG Time-To-First-Token (ms) and Multi-Hop F1 on 70B LLM (synthetic + anonymized_reported)",
    "encodings": {
      "x": "tier",
      "y": "ttft_ms",
      "color": "tier"
    }
  }
}
```

---

## ⑥ 🛠️ Production Verification Checklist (Good fit vs. Not for)

### ✅ Good fit vs. ⛔ Not for
- **✅ Good fit**: Enterprise RAG pipelines, dynamic MCP tool-schema assembly, and modular few-shot prompting.
- **⛔ Not for**: Concatenating independently cached KV blocks without `RoPE` offset correction and sparse cross-attention recomputation.

### Production Checklist
- [ ] **Apply relative `RoPE` rotation $R(p_{\text{new}} - p_{\text{old}})$ when loading non-prefix KV chunks**: Align positional encodings to exact runtime offsets.
- [ ] **Recompute the top `15%` high-drift tokens identified at Layer 1 (`CacheBlend`)**: Use `Gather-Compact-Scatter` to preserve dense Tensor Core efficiency and multi-hop accuracy.
- [ ] **Overlap Host-to-Device KV streaming with `15%` sparse recomputation on separate CUDA streams**: Hide PCIe transfer latency completely.
