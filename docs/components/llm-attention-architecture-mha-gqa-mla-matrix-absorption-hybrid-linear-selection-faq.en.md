# LLM Attention & KV Cache Architecture Evolution: `MHA` vs. `GQA-8` vs. `MLA (Multi-head Latent Attention & Weight Matrix Absorption)` vs. `7:1 Hybrid Linear Attention` Guide

> **Language**: [中文](./llm-attention-architecture-mha-gqa-mla-matrix-absorption-hybrid-linear-selection-faq.md) | **English (Current)**  
> **Audience**: Cloud & AI Infrastructure Solution Architects and LLM Serving Architects  
> **Provenance**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)

![LLM Attention Evolution MHA vs GQA vs MLA Matrix Absorption Summary](../../assets/svg/llm-attention-architecture-mha-gqa-mla-matrix-absorption-hybrid-linear-selection-faq-summary.svg)

---

## 1. 📌 Executive Summary

> **Bottom Line**: **Long-context LLM serving is bottlenecked by linear KV Cache growth: evolving from `MHA` (`1,310.7 KB/tok` FP8) to `GQA-8` (`163.8 KB/tok`, `8x` smaller) and `DeepSeek-V2/V3/R1`'s `MLA` (`Multi-head Latent Attention`, caching only a `d_c=512` compressed latent vector + `d_r=64` decoupled RoPE key = **`35.14 KB/tok` FP8**, `4.67x` smaller than GQA-8 and `37.3x` smaller than MHA!). MLA's decisive engineering breakthrough is Decode-time Weight Matrix Absorption—absorbing $W^{UK}$ into $W^Q$ and $W^{UV}$ into $W^O$ so all `128` Query heads attend directly to the single shared compressed latent vector $c_t^{KV}$ without decompressing K/V in HBM. This expands `32K` concurrency on an `8x H200` node from `72` to `236` sessions (`3.28x`), raises Decode arithmetic intensity from `16` to `256 FLOPs/Byte` (`66.8%` TensorCore utilization via `FlashMLA`), and cuts `$/1M` tokens by `66.7%–92.7%`!**

- **Interactive Attention KV & Arithmetic Intensity Sizer**: [Open Interactive Sizer (`llm-attention-architecture-mha-gqa-mla-matrix-absorption-hybrid-linear-sizer.html`)](../reports/llm-attention-architecture-mha-gqa-mla-matrix-absorption-hybrid-linear-sizer.html)
- **7-Card 4:5 Visual Report**: [Open 7-Card Report (`llm-attention-architecture-mha-gqa-mla-matrix-absorption-hybrid-linear-selection-cards.html`)](../reports/llm-attention-architecture-mha-gqa-mla-matrix-absorption-hybrid-linear-selection-cards.html)

---

## 2. 📊 4-Tier Attention Comparison Matrix (`8x H200` @ `32K` Context)

| Metric | Tier 1: Legacy `MHA-64` | Tier 2: Standard `GQA-8` (`Llama-3.3-70B`) | Tier 3: Naive Unabsorbed `MLA` Decode | Tier 4 (Recommended): Absorbed `MLA (FlashMLA)` / `7:1 Hybrid Linear` |
| :--- | :--- | :--- | :--- | :--- |
| **FP8 KV Cache per Token** | `1,310.7 KB/tok` ✕ | `163.8 KB/tok` | `35.1 KB/tok` | **`35.1 KB/tok` ✓ (`-4.67x` vs. GQA)** |
| **KV Cache per `32K` Session** | `40.00 GB` ✕ | `5.00 GB` | `1.10 GB` | **`1.10 GB` ✓** |
| **Max `32K` Sessions per `8x H200`** | `9` ✕ | `72` | `64` | **`236` ✓ (`3.28x` vs. GQA)** |
| **Decode TensorCore Utilization** | `6.2%` | `14.5%` | `19.0%` | **`66.8%` ✓ (`AI = 256 FLOPs/B`)** |
| **Cost per 1M `32K` Tokens** | `$3.85 / 1M` | `$0.84 / 1M` | `$1.15 / 1M` | **`$0.28 / 1M` ✓ (`-92.7%`)** |

---

## 3. 🛠️ Production Readiness Checklist

- [ ] **Verify `FlashMLA` + Weight Matrix Absorption**: Ensure `vLLM / SGLang` executes absorbed latent MQA kernels during Decode rather than decompressing `c_KV` into 128 K/V heads.
- [ ] **Leverage `MLA` to Slash PD Disaggregation RDMA Bandwidth by `4.67x`**.
