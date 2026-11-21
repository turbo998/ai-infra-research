# Speculative Decoding Deep Dive & Selection Boundary: Standalone Draft vs. `Medusa` vs. `EAGLE-3` vs. `DeepSeek MTP`, Acceptance Math $\tau(\alpha,\gamma)$ & High-Batch Crossover Guide

> **Language**: [中文](./speculative-decoding-draft-medusa-eagle3-mtp-acceptance-batch-boundary-selection-faq.md) | **English (Current)**  
> **Audience**: Cloud & AI Infrastructure Solution Architects and LLM Serving Optimization Leads  
> **Provenance**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)

![Speculative Decoding Draft vs Medusa vs EAGLE-3 vs MTP Summary](../../assets/svg/speculative-decoding-draft-medusa-eagle3-mtp-acceptance-batch-boundary-selection-faq-summary.svg)

---

## 1. 📌 Executive Summary

> **Bottom Line**: **Speculative decoding converts idle Decode TensorCore FLOPs into HBM bandwidth savings by verifying $\gamma + 1$ candidate tokens in a single target-model weight read with 100% mathematically lossless rejection sampling: $\tau(\alpha, \gamma) = \frac{1 - \alpha^{\gamma+1}}{1 - \alpha}$ and $\text{Speedup} = \frac{\tau(\alpha, \gamma)}{1 + \gamma \cdot c_{\text{draft}} + c_{\text{verify}}(B, \gamma)}$. Replace heavy standalone `7B` draft models ($c_{\text{draft}} \approx 11.5\%$) and context-independent `Medusa` heads with native `DeepSeek MTP` or `EAGLE-3` feature-extrapolation trees ($c_{\text{draft}} < 1.8\%, \tau = 3.12\text{ tok/step}$, cutting effective TPOT from `21.8ms` to `8.9ms`, `2.45x` faster), and enforce an Adaptive Batch Cutoff (`disable_by_batch_size = 48`) so high-concurrency batches never cross the Roofline ridge point into negative speedup!**

- **Interactive Speculative Speedup & Batch Crossover Calculator**: [Open Interactive Sizer (`speculative-decoding-draft-medusa-eagle3-mtp-acceptance-batch-boundary-sizer.html`)](../reports/speculative-decoding-draft-medusa-eagle3-mtp-acceptance-batch-boundary-sizer.html)
- **7-Card 4:5 Visual Report**: [Open 7-Card Report (`speculative-decoding-draft-medusa-eagle3-mtp-acceptance-batch-boundary-selection-cards.html`)](../reports/speculative-decoding-draft-medusa-eagle3-mtp-acceptance-batch-boundary-selection-cards.html)

---

## 2. 📊 4-Tier Speculative Decoding Comparison Matrix (`8x H200`)

| Metric | Tier 1: Standard Autoregressive (`1.0 tok/step`) | Tier 2: Standalone `7B` Draft Model (Always-On) | Tier 3: Static `Medusa` Parallel Heads | Tier 4 (Recommended): Native `MTP` / `EAGLE-3` + Adaptive Batch Cutoff |
| :--- | :--- | :--- | :--- | :--- |
| **Draft Overhead ($c_{\text{draft}}$)** | `0.0%` | `11.5%` ✕ | `1.5%` | **`1.8%` ✓ (1-Layer Feature Head)** |
| **Accepted Tokens / Step ($\tau$)** | `1.00` | `2.35` | `1.85` | **`3.12` ✓ (`alpha = 0.82–0.85`)** |
| **Effective P99 `TPOT`** | `21.8 ms` | `14.2 ms` | `13.4 ms` | **`8.9 ms` ✓ (`2.45x` faster)** |
| **High-Batch (`B=96`) TPS / GPU** | `1,680 tok/s` | `1,190 tok/s` ✕ (`-29%` inversion!) | `1,520 tok/s` | **`2,290 tok/s` ✓ (Auto-throttled)** |
| **Cost per 1M Tokens** | `$0.62 / 1M` | `$0.58 / 1M` | `$0.49 / 1M` | **`$0.29 / 1M` ✓ (`-53.2%`)** |

---

## 3. 🛠️ Production Readiness Checklist

- [ ] **Adopt Native `MTP` or `EAGLE-3` Over Standalone Draft Models** to keep draft cost `<2%` and share the target model's TP layout.
- [ ] **Configure `--speculative-disable-by-batch-size 48`** to prevent compute-bound slowdown during traffic spikes.
