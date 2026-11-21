# Ultra-Long Context (`128K–1M`) LLM Training & Serving Architecture: `2D Hybrid USP (Ulysses × Ring CP)`, 3-Tier KV Cache Offload (`HBM -> Host DRAM -> NVMe GDS`) & Prefix Routing Guide

> **Language**: [中文](./long-context-128k-1m-ring-ulysses-cp-kv-offload-prefix-routing-selection-faq.md) | **English (Current)**  
> **Audience**: Cloud & AI Infrastructure Solution Architects, Long-Context Serving Leads, and Distributed Systems Engineers  
> **Provenance**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)

![Ultra-Long Context 128K-1M 2D USP & Tiered KV Offload Summary](../../assets/svg/long-context-128k-1m-ring-ulysses-cp-kv-offload-prefix-routing-selection-faq-summary.svg)

---

## 1. 📌 Executive Summary

> **Bottom Line**: **When sequence length $S$ scales beyond `64K–128K` toward `1M` tokens, LLMs hit two simultaneous walls: (1) quadratic self-attention FLOPs ($\mathcal{O}(S^2)$) overtake linear projection FLOPs once $S > 12 d_{\text{model}} \approx 96\text{K}$, pushing single-node `TP=8` `128K` TTFT to `28.4s`; and (2) single-session KV Cache reaches `4.5–21.0 GB`, causing frequent HBM eviction and catastrophic $\mathcal{O}(S^2)$ recomputation. Combining 2D Hybrid Unified Sequence Parallelism (`Intra-Node Ulysses [NVLink All-to-All] × Inter-Node Ring Attention [RDMA P2P]`) with a 3-Tier KV Offload Pool (`L1 GPU HBM <-> L2 Pinned Host DDR5 <-> L3 NVMe GDS / Mooncake RDMA Pool`) and Prefix-Hash Routing cuts `128K` P95 TTFT from `28.4s` to `2.3s` (`12.3x` faster), raises multi-turn KV hit rate to `91.4%`, and reduces cost per 1M tokens by `82.9%` (`$0.58/1M`)!**

- **Interactive `128K–1M` CP & KV Offload Calculator**: [Open Interactive Sizer (`long-context-128k-1m-ring-ulysses-cp-kv-offload-prefix-routing-sizer.html`)](../reports/long-context-128k-1m-ring-ulysses-cp-kv-offload-prefix-routing-sizer.html)
- **7-Card 4:5 Visual Report**: [Open 7-Card Report (`long-context-128k-1m-ring-ulysses-cp-kv-offload-prefix-routing-selection-cards.html`)](../reports/long-context-128k-1m-ring-ulysses-cp-kv-offload-prefix-routing-selection-cards.html)

---

## 2. 🎯 Architect Deep-Dive Q&A

### Q1: Why does `2D Hybrid USP (Ulysses × Ring)` beat pure `DeepSpeed-Ulysses` or pure `Ring Attention` for `128K–1M` contexts?
- **DeepSpeed-Ulysses** uses `All-to-All` to partition attention heads across GPUs, but is strictly capped by the number of KV heads ($n_{kv} = 8$ in GQA-8 models like `Llama-3.3-70B/405B`).
- **Ring Attention** circulates KV blocks via `P2P Send/Recv` overlapped with `FlashAttention` without head-count limits, but suffers multi-hop latency across many nodes.
- **2D Hybrid USP** runs `Ulysses=8` inside each 8-GPU `900 GB/s NVLink` node and `Ring=2/4/8` across nodes over `400G/800G RDMA` with ZigZag causal load balancing.

### Q2: Why is recalling `21 GB` of 128K KV from Host DRAM (`41ms`) `56–692x` faster than recomputing Prefill?
At `512 GB/s` aggregate PCIe Gen5 bandwidth per 8-GPU node (`8 × 64 GB/s`), streaming `21.0 GB` of warm KV blocks back into GPU HBM takes only **`41.0 ms`**, compared to **`2,300–28,400 ms`** to recompute `128K` quadratic attention!

---

## 3. 📊 4-Tier Ultra-Long Context (`128K`) Comparison Matrix

| Metric | Tier 1: Single-Node `TP=8` Only (Full Recompute) | Tier 2: Pure `Ring Attention CP` | Tier 3: `Ulysses CP` + Local Host DRAM | Tier 4 (Recommended): `2D Hybrid USP` + Distributed `DRAM/GDS KV Pool` |
| :--- | :--- | :--- | :--- | :--- |
| **`128K` P95 `TTFT`** | `28.4 s` ✕ | `9.8 s` | `6.4 s` | **`2.3 s` ✓ (`12.3x` faster)** |
| **`128K` P99 `TPOT`** | `62.5 ms` ✕ | `29.4 ms` | `21.2 ms` ✓ | **`15.6 ms` ✓** |
| **Multi-Turn KV Cache Hit Rate** | `18.5%` ✕ | `34.0%` | `68.5%` | **`91.4%` ✓** |
| **Cost per 1M `128K` Tokens** | `$3.40 / 1M` ✕ | `$1.65 / 1M` | `$1.02 / 1M` | **`$0.58 / 1M` ✓ (`-82.9%`)** |

---

## 4. 🛠️ Production Readiness Checklist

- [ ] **Enable `2D Hybrid USP (Ulysses=8 × Ring)`** with ZigZag causal load balancing for `>=64K` contexts.
- [ ] **Deploy Tiered KV Offloading (`LMCache / Mooncake` over `1.5–2 TB` Pinned Host DDR5 + `NVMe GDS`)** so evicted `128K` sessions recall in `<45ms` instead of recomputing.
- [ ] **Enforce Prefix-Hash Consistent Routing** at the AI Gateway.
