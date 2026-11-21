# Hardware-Aligned Sparse Attention Guide (`128K–1M` Context): `DeepSeek NSA (Native Sparse Attention)` vs. `Moonshot MoBA` vs. `Dense FlashAttention-3`

> **Language**: [中文](./sparse-attention-nsa-moba-block-sparse-flashattention3-long-context-selection-faq.md) | **English (Current)**  
> **Audience**: Cloud & AI Infrastructure Solution Architects and Long-Context Foundation Model Engineers  
> **Provenance**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)

![Hardware-Aligned Sparse Attention DeepSeek NSA vs Moonshot MoBA vs FA-3 Summary](../../assets/svg/sparse-attention-nsa-moba-block-sparse-flashattention3-long-context-selection-faq-summary.svg)

---

## 1. 📌 Executive Summary

> **Bottom Line**: **At `128K–1M` context lengths, quadratic $O(S^2)$ Dense `FlashAttention-3` (`Tier 1`) still takes `11.80 seconds` of Prefill TTFT (`42.0 ms` Decode TPOT) on `8x H100`, while static heuristic sliding windows (`StreamingLLM`, `Tier 2`) collapse to `24.5%` recall on middle-document facts. Modern long-context infrastructure relies on Hardware-Aligned Trainable Block-Sparse Attention: (1) `Moonshot MoBA (Mixture of Block Attention)` (`Tier 3`), which uses parameter-free mean-pooled block keys and `flash_attn_varlen` to accelerate `128K` Prefill by `4.82x` (`2.45s` TTFT, `99.1%` RULER accuracy) as a drop-in upgrade for existing dense checkpoints; and (2) `DeepSeek NSA (Native Sparse Attention)` (`Tier 4`), which combines coarse token compression, GQA-group-shared contiguous fine block selection, and a local sliding window to cut `128K` Prefill TTFT `9.01x` (`11.80s -> 1.31s`), accelerate `128K` Decode TPOT `4.2x` (`42.0ms -> 10.0ms`), and achieve `99.7%` RULER-128K accuracy.**

- **Interactive Sparse Attention (`NSA / MoBA / FA-3`) Sizer**: [Open Interactive Sizer (`sparse-attention-nsa-moba-block-sparse-flashattention3-long-context-sizer.html`)](../reports/sparse-attention-nsa-moba-block-sparse-flashattention3-long-context-sizer.html)
- **7-Card 4:5 Visual Report**: [Open 7-Card Report (`sparse-attention-nsa-moba-block-sparse-flashattention3-long-context-selection-cards.html`)](../reports/sparse-attention-nsa-moba-block-sparse-flashattention3-long-context-selection-cards.html)

---

## 2. 📊 4-Tier `128K` Long-Context Attention Matrix (`70B Model` · `128K Context` · `8x H100`)

| Metric | Tier 1: Dense `FlashAttention-3` ($O(S^2)$) | Tier 2: Static Sliding Window + Sinks (`StreamingLLM`) | Tier 3: `Moonshot MoBA` (Parameter-Free Block Router) | Tier 4 (Recommended): `DeepSeek NSA` (3-Branch Hardware-Aligned) |
| :--- | :--- | :--- | :--- | :--- |
| **`128K` Prefill TTFT** | `11.80 sec` ✕ (`1.00x`) | `1.45 sec` (`8.14x`) | **`2.45 sec` ✓ (`4.82x` faster)** | **`1.31 sec` ✓ (`9.01x` faster)** |
| **`128K` Decode TPOT** | `42.0 ms` ✕ | `8.5 ms` | `18.5 ms` | **`10.0 ms` ✓ (`4.2x` faster)** |
| **`RULER-128K` Benchmark Recall** | `98.8%` | `24.5%` ✕ (Middle lost) | **`99.1%` ✓** | **`99.7%` ✓ (Beats dense!)** |
| **Architecture Fit** | Short/medium `<32K` | Casual streaming chat | **Drop-in CPT on dense models ✓** | **Native `128K–1M` foundation models ✓** |

---

## 3. 🛠️ Production Readiness Checklist

- [ ] **Enforce GQA Group-Shared Contiguous Block Selection (`64–128` tokens/block)** in sparse decode kernels so all Query heads sharing a GQA KV head fetch the exact same KV blocks from HBM into SRAM once.
- [ ] **Use `MoBA` When Retrofitting Existing Pre-Trained Dense Models to `128K–512K` Context**, and **`NSA` When Pre-Training Native Long-Context Reasoning Models**.
