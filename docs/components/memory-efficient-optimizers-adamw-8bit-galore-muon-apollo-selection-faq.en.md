# Memory-Efficient & Orthogonalized Optimizers Guide: `FP32 AdamW (12Ψ)` vs. `8-Bit AdamW` vs. `GaLore / APOLLO` vs. `Distributed Muon (Newton-Schulz)`

> **Language**: [中文](./memory-efficient-optimizers-adamw-8bit-galore-muon-apollo-selection-faq.md) | **English (Current)**  
> **Audience**: Cloud & AI Infrastructure Solution Architects and Distributed Training Engineers  
> **Provenance**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)

![Memory-Efficient Optimizers & Distributed Muon Summary](../../assets/svg/memory-efficient-optimizers-adamw-8bit-galore-muon-apollo-selection-faq-summary.svg)

---

## 1. 📌 Executive Summary

> **Bottom Line**: **In full-parameter `70B` LLM training, standard `FP32 AdamW` (`Tier 1`) consumes `12 Bytes/parameter` (`4B` master weights + `4B` $m_t$ + `4B` $v_t$ = `840.0 GB` total state, `6x` larger than the `140 GB` BF16 weights!). Modern training platforms optimize along two axes: (1) For HBM-constrained full-parameter SFT/CPT, `Blockwise 8-Bit AdamW` (`Tier 2`, `420.0 GB`) and `GaLore / APOLLO Low-Rank Gradient Projection` (`Tier 3`, `168.0 GB` — `-80%` state HBM) eliminate memory bottlenecks without freezing rank adapters; (2) For large-scale pre-training and continual pre-training, `Distributed Muon (MomentUm Orthogonalized by Newton-Schulz)` (`Tier 4`, proven in `Kimi Moonlight`) applies 5 fast BF16 TensorCore Newton-Schulz iterations to orthogonalize 2D weight momentum while dropping second-moment $v_t$ (`560.0 GB` / `8Ψ` state)—reaching the same target validation loss in `1.42x` fewer training tokens and cutting total cluster GPU hours by `29.5%`.**

- **Interactive Optimizer HBM & Convergence Sizer**: [Open Interactive Sizer (`memory-efficient-optimizers-adamw-8bit-galore-muon-apollo-sizer.html`)](../reports/memory-efficient-optimizers-adamw-8bit-galore-muon-apollo-sizer.html)
- **7-Card 4:5 Visual Report**: [Open 7-Card Report (`memory-efficient-optimizers-adamw-8bit-galore-muon-apollo-selection-cards.html`)](../reports/memory-efficient-optimizers-adamw-8bit-galore-muon-apollo-selection-cards.html)

---

## 2. 📊 4-Tier Optimizer Architecture Matrix (`70B Model` · `Target Validation Loss = 1.85`)

| Metric | Tier 1: Standard `FP32 AdamW` (`12Ψ`) | Tier 2: Blockwise `8-Bit AdamW` (`6Ψ`) | Tier 3: `GaLore / APOLLO` Low-Rank (`2.4Ψ`) | Tier 4 (Recommended): `Distributed Muon` (`8Ψ`, Newton-Schulz 2D) |
| :--- | :--- | :--- | :--- | :--- |
| **Optimizer Bytes / Parameter** | `12.0 B / param` ✕ | `6.0 B / param` | **`2.4 B / param` ✓ (`-80%`)** | **`8.0 B / param` ✓ (No 2D $v_t$)** |
| **`70B` Total Optimizer State** | `840.0 GB` ✕ | `420.0 GB` | **`168.0 GB` ✓** | **`560.0 GB` ✓ (`-280 GB` vs Tier 1)** |
| **Token Sample Efficiency** | `1.00x` | `1.00x` | `0.98x` | **`1.42x` ✓ (`30%` fewer tokens needed)** |
| **Total GPU Hours Saved** | `0.0%` | `8.5%` | `18.0%` | **`29.5%` ✓ (`-$147.5k` on `$500k` run)** |

---

## 3. 🛠️ Production Readiness Checklist

- [ ] **Restrict `Muon` Orthogonalization Strictly to Internal 2D Linear Matrices (`QKV, O, MLP`)** while keeping `Embedding`, `LM_Head`, and `1D RMSNorm` parameters on `AdamW`.
- [ ] **Include Weight Decay and `0.2 * sqrt(max(d_out, d_in))` RMS Update Scaling in Distributed `Muon`** so learning rates transfer cleanly across model widths and long-horizon training remains bounded.
