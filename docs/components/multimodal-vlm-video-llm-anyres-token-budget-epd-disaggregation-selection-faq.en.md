# Multimodal VLM & Video-LLM Serving Architecture Selection: Dynamic-Resolution (`NaViT / AnyRes`) Token Math, 3-Stage `E-P-D` (`Encoder-Prefill-Decode`) Disaggregation & Visual Hash Cache Guide

> **Language**: [中文](./multimodal-vlm-video-llm-anyres-token-budget-epd-disaggregation-selection-faq.md) | **English (Current)**  
> **Audience**: Cloud & AI Infrastructure Solution Architects and Multimodal Platform Leads  
> **Provenance**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)

![Multimodal VLM AnyRes Token Budget & 3-Stage E-P-D Disaggregation Summary](../../assets/svg/multimodal-vlm-video-llm-anyres-token-budget-epd-disaggregation-selection-faq-summary.svg)

---

## 1. 📌 Executive Summary

> **Bottom Line**: **Multimodal VLM serving (`Qwen2.5-VL-72B`, `InternVL-78B`) faces two severe bottlenecks: (1) Visual Token Explosion—at `14×14` ViT patches with a `2×2` spatial merger (`28×28 = 784 pixels/token`), a single uncapped `4K (3840×2160)` image spawns **`10,764` visual tokens** (`1.76 GB` GQA KV Cache!); and (2) Stage Interference—colocating the `0.6B ViT Encoder` (zero KV Cache, pure compute), `72B LLM Prefill` (compute-bound), and `72B LLM Decode` (HBM bandwidth-bound) on the same GPU inflates P99 Decode TPOT to `68.4ms`. Enforcing task-specific `max_pixels` budgets, decoupling into a 3-Stage `E-P-D` pipeline (`L40S ViT Pool -> H100 Prefill Pool -> H200 Decode Pool`), and caching `SHA-256` visual embeddings/KV prefixes cuts P95 TTFT `8.0x` (`480ms`), stabilizes P99 TPOT at `15.2ms`, and reduces cost per 1,000 multimodal queries by `78.2%` (`$14.20 -> $3.10`)!**

- **Interactive Visual Token & `E-P-D` Sizer**: [Open Interactive Sizer (`multimodal-vlm-video-llm-anyres-token-budget-epd-disaggregation-sizer.html`)](../reports/multimodal-vlm-video-llm-anyres-token-budget-epd-disaggregation-sizer.html)
- **7-Card 4:5 Visual Report**: [Open 7-Card Report (`multimodal-vlm-video-llm-anyres-token-budget-epd-disaggregation-selection-cards.html`)](../reports/multimodal-vlm-video-llm-anyres-token-budget-epd-disaggregation-selection-cards.html)

---

## 2. 📊 4-Tier Multimodal VLM Serving Comparison Matrix (`72B VLM`)

| Metric | Tier 1: Monolithic Colocated (`ViT+P+D`) + Uncapped 4K | Tier 2: Colocated + `max_pixels` Cap (`2,048` tok) | Tier 3: 2-Stage P/D Split (`ViT` on `H200`) | Tier 4 (Recommended): 3-Stage `E-P-D` Split (`L40S + H100 + H200`) + Visual Cache |
| :--- | :--- | :--- | :--- | :--- |
| **P95 `TTFT`** | `3,850 ms` ✕ | `1,240 ms` | `820 ms` | **`480 ms` ✓ (`8.0x` faster)** |
| **P99 Decode `TPOT`** | `68.4 ms` ✕ | `36.2 ms` ✕ | `17.8 ms` ✓ | **`15.2 ms` ✓ (`4.5x` faster)** |
| **Goodput RPS / Node** | `2.4 RPS` | `6.8 RPS` | `11.2 RPS` | **`16.5 RPS` ✓ (`6.88x`)** |
| **Cost per 1,000 Queries** | `$14.20 / 1K` | `$6.40 / 1K` | `$4.95 / 1K` | **`$3.10 / 1K` ✓ (`-78.2%`)** |

---

## 3. 🛠️ Production Readiness Checklist

- [ ] **Enforce Task-Adaptive `max_pixels` Caps** (`1,280` visual tokens for chat/UI; `4,096` for dense OCR).
- [ ] **Disaggregate into 3-Stage `E-P-D` Pools** so stateless `ViT` encoders run on inexpensive `L40S` GPUs rather than `141GB H200` nodes.
- [ ] **Enable Content-Addressable `SHA-256` Visual Embedding & Radix KV Caching** for multi-turn visual sessions.
