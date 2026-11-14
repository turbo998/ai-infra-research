# Million-Token Long-Context Parallelism (`Megatron SP` vs. `DeepSpeed Ulysses` vs. `ZigZag Ring` vs. `Hybrid 2D USP`) Selection Guide

> **Language / 语言**: [中文](./long-context-parallelism-tp-sp-ulysses-zigzag-ring-hybrid-usp-selection-faq.md) | **English (Current)**  
> **End-to-End Interactive Artifacts**: [🧮 Interactive Long-Context Parallelism Sizer](../reports/long-context-parallelism-tp-sp-ulysses-zigzag-ring-hybrid-usp-sizer.html) · [🃏 7-Card 4:5 Visual Report](../reports/long-context-parallelism-tp-sp-ulysses-zigzag-ring-hybrid-usp-selection-cards.html) · [🧪 Sealed Evaluator & Data](../../experiments/long_context_parallelism_ulysses_ring_usp_selection/run.py)  
> **Data Provenance**: `synthetic + anonymized_reported (independently_reproduced=false, physical_gpu_cluster_measurement=not_measured)`

![Million-Token Long-Context Parallelism (Megatron SP vs. Ulysses vs. ZigZag Ring vs. Hybrid USP) Selection Summary](../../assets/svg/long-context-parallelism-tp-sp-ulysses-zigzag-ring-hybrid-usp-selection-faq-summary.svg)

---

## 🔥 Daily Hottest Capability Breakthrough

- **Breakthrough Focus**: **Unifying `512K–4M` Long-Context Distributed Training and Prefill Parallelism—Demystifying the Architectural Boundaries of `Megatron-LM Sequence Parallelism (SP)`, `DeepSpeed Ulysses` (All-to-All Head Sharding), and `ZigZag Ring Attention` (Symmetric Head-Tail Chunk Pairing), and Achieving `54.2%` 32-GPU MFU (`2.46x` Faster than Naive Ring) via `Hybrid 2D USP` (`Intra-Node NVLink Ulysses=8` × `Inter-Node RDMA ZigZag Ring=4`)**.
- **Why It Matters Today**: At `512K–4M` context lengths, activations and KV Cache exceed single-GPU and even single-node HBM capacity. Yet teams frequently hit three major pitfalls: (1) `Megatron SP` still `AllGather`s the full sequence before Attention; (2) scaling pure `DeepSpeed Ulysses` beyond `8` GPUs on a 70B GQA model (`8` KV heads) forces `4x` KV head replication and cross-node `All-to-All` congestion (`31.5%` MFU); and (3) naive contiguous `Ring Attention` suffers a **`7.0x` causal triangle load imbalance** (`22.0%` MFU)!

---

## 📌 1. Executive Summary (3-Minute Architect Takeaways)

1. **Clear Distinction Between `Megatron SP`, `Ulysses`, and `Ring Attention`**:
   - **`Megatron TP + SP`**: Shards sequence `[L/TP, B, H]` only across `RMSNorm / Dropout`, then `AllGather`s back to full length `[L, B, H/TP]` before Attention. Cannot scale beyond 1 node (`TP=8`).
   - **`DeepSpeed Ulysses` (`Tier 2`)**: Uses `All-to-All` before Attention to transpose `[L/P, B, H]` into `[L, B, H/P]` (`4x` lower communication volume than Ring Attention!), **but is strictly bounded by the number of GQA KV heads (`8` on 70B models)**.
   - **`ZigZag Ring Attention` (`Tier 3`)**: Shards sequence across any number of GPUs while passing KV chunks around a P2P ring and merging block outputs via online `logaddexp(LSE)`. Pairing `Chunk(r)` with `Chunk(2R - 1 - r)` (**ZigZag folding**) eliminates the `7.0x` causal mask load imbalance (`1.0x` balance, `42.8%` MFU) ✅.
2. **Production Gold Standard (`Tier 4`: `Hybrid 2D USP` = Intra-Node NVLink `Ulysses=8` × Inter-Node RDMA `ZigZag Ring=4`)**:
   - Maps **`Ulysses = 8`** onto intra-node `900 GB/s` NVLink (matching the `8` GQA KV heads with zero KV duplication) and **`ZigZag Ring = 4`** onto inter-node `400G/800G` RDMA (reducing cross-node ring steps from `32` to `4` large chunks), cutting exposed communication bubble to **`4.2%`** and reaching **`54.2%` MFU (`2.46x` faster)** 🏆!

---

## 🎯 2. Architect & Customer Deep-Dive Q&A (Q1 / Q2 / Q3)

### Q1: Why does `DeepSpeed Ulysses` transmit `4x` less data per GPU than `Ring Attention` inside a node?
- **Answer**: Across 4 `All-to-All` transposes (`Q, K, V` forward + `O` reverse), `Ulysses` sends $\frac{4 L d}{P}$ bytes per GPU (which decreases with $P$), whereas `Ring Attention` circulates $K, V$ across $P-1$ steps for $\approx 2 L d$ bytes per GPU. At $P = 8$, `Ulysses` sends $0.5 L d$ vs. Ring's $2.0 L d$ (**`4x` less traffic**).

### Q2: How does `Ring Attention` combine partial `FlashAttention` outputs across ring steps without precision loss?
- **Answer**: Using the per-row log-sum-exp (`softmax_lse`) returned by `FlashAttention`:
  $$\text{LSE}_{\text{new}} = \text{logaddexp}(\text{LSE}_{\text{acc}}, \text{LSE}_{\text{step}}), \quad O_{\text{acc}} = O_{\text{acc}} e^{\text{LSE}_{\text{acc}} - \text{LSE}_{\text{new}}} + O_{\text{step}} e^{\text{LSE}_{\text{step}} - \text{LSE}_{\text{new}}}$$
  which is mathematically identical to full-sequence attention.

### Q3: What is the critical `RoPE` pitfall when enabling `ZigZag` sequence partitioning?
- **Answer**: Because `Rank r` holds two non-contiguous slices (`Chunk(r)` at the beginning and `Chunk(2R-1-r)` near the end), you must pass the corresponding permuted **`position_ids`** into `RoPE` and the causal mask generator.

---

## 🧭 3. Visual Architecture Diagrams (Mermaid)

```mermaid
flowchart TD
    A["512K - 4M Long-Context Parallelism Selection"] --> B{"Single node (<=8 GPUs) and P <= num_kv_heads?"}
    B -- "Yes" --> U8["✅ Intra-Node DeepSpeed Ulysses (U=8)<br/>4x lower comm volume over 900 GB/s NVLink"]
    B -- "No (Multi-Node 16-128+ GPUs)" --> C{"Sequence partitioning & collective topology?"}
    C -- "Contiguous Ring [0..L/P]" --> T1["⛔ Tier 1: Contiguous Ring Attention<br/>7.0x causal triangle load imbalance, 22.0% MFU ✕"]
    C -- "Pure 32-GPU Ulysses" --> T2["🟨 Tier 2: Pure Multi-Node Ulysses<br/>4x GQA KV replication + cross-node All-to-All, 31.5% MFU ⚠️"]
    C -- "Pure 1D ZigZag Ring (32 steps)" --> T3["🟦 Tier 3: Pure 1D ZigZag Ring Attention<br/>1.0x load balance, 19.5% comm bubble, 42.8% MFU ✅"]
    C -- "2D Hybrid USP (Recommended)" --> T4["🏆 Tier 4: Hybrid 2D USP (NVLink U=8 x RDMA ZigZag R=4)<br/>Zero KV replication + 4 large ring steps, 4.2% bubble, 54.2% MFU"]
```

---

## 📊 4. Architect Decision Matrix (4 Tiers Compared)

| Dimension | 🟥 Tier 1: Contiguous `Ring` | 🟨 Tier 2: Pure 32-GPU `Ulysses` | 🟦 Tier 3: Pure 1D `ZigZag Ring` | 🟩 Tier 4: `Hybrid 2D USP (U=8 × R=4)` (Recommended 🏆) |
| :--- | :--- | :--- | :--- | :--- |
| **Causal Load Imbalance Ratio** | `7.0x` ✕ | **`1.0x`** ✅ | **`1.0x`** ✅ | **`1.0x` (Perfect Balance)** 🏆 |
| **Exposed Comm Bubble (`%`)** | `44.5%` ✕ | `36.0%` ⚠️ | `19.5%` | **`4.2%`** 🏆 |
| **32-GPU 512K Cluster MFU** | `22.0%` ✕ | `31.5%` | `42.8%` ✅ | **`54.2%` (`2.46x` faster)** 🏆 |
| **Max Supported Context** | `256K` | `512K` | `2,048K (2M)` ✅ | **`4,096K (4M Tokens)`** 🏆 |

---

## 🧪 5. Quantitative Case Study & 7-Card 4:5 Benchmark Report

```chart
{
  "data": {
    "values": [
      { "tier": "Tier 1: Naive Contiguous Ring (7x Imbalance)", "mfu_pct": 22.0, "bubble_pct": 44.5 },
      { "tier": "Tier 2: Pure 32-GPU Ulysses (4x KV Dup)", "mfu_pct": 31.5, "bubble_pct": 36.0 },
      { "tier": "Tier 3: Pure 1D ZigZag Ring (32 Steps)", "mfu_pct": 42.8, "bubble_pct": 19.5 },
      { "tier": "Tier 4: Hybrid 2D USP (U=8 NVLink x R=4 RDMA)", "mfu_pct": 54.2, "bubble_pct": 4.2 }
    ]
  },
  "semantic_types": {
    "tier": "Category",
    "mfu_pct": "Percentage",
    "bubble_pct": "Percentage"
  },
  "chart_spec": {
    "chartType": "Bar Chart",
    "title": "Hybrid 2D USP (Ulysses=8 x ZigZag Ring=4) boosts 32-GPU 512K MFU from 22.0% to 54.2%",
    "subtitle": "32-GPU Model FLOPs Utilization (MFU %) at 512K context length across 4 parallelism tiers",
    "encodings": {
      "x": "tier",
      "y": "mfu_pct"
    }
  }
}
```

- 🧮 **Interactive Long-Context Parallelism Sizer**: [`docs/reports/long-context-parallelism-tp-sp-ulysses-zigzag-ring-hybrid-usp-sizer.html`](../reports/long-context-parallelism-tp-sp-ulysses-zigzag-ring-hybrid-usp-sizer.html)
- 🃏 **7-Card 4:5 Visual Benchmark Report**: [`docs/reports/long-context-parallelism-tp-sp-ulysses-zigzag-ring-hybrid-usp-selection-cards.html`](../reports/long-context-parallelism-tp-sp-ulysses-zigzag-ring-hybrid-usp-selection-cards.html)

---

## 🛠️ 6. Production Verification Checklist

- [ ] **Cap `ulysses_degree <= min(8, num_key_value_heads)`**: Keep `Ulysses` `All-to-All` inside the intra-node NVLink domain without GQA KV head duplication.
- [ ] **Enforce `ZigZag` (`Chunk(r) + Chunk(2R-1-r)`) Pairing for Causal Masks**: Eliminate the `7.0x` causal triangle compute imbalance across ring ranks.
- [ ] **Permute `position_ids` Alongside ZigZag Token Chunks**: Ensure `RoPE` applies exact document positions to both the head chunk and tail chunk.
- [ ] **Accumulate Online `softmax_lse` in `FP32`**: Preserve bitwise accuracy across ring steps.
