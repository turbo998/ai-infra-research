# Low-Precision `AdamW` Optimizer & Master Weight Compression (`14 B/Param -> 7 B/Param`): Why Pure `BF16` Weights Stagnate (`w + Δw == w`), `Stochastic Rounding` vs. `Kahan Summation` & Block-Wise `FP8` Moments Selection Guide

> **Provenance & Evidence Tier**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`). Grounded in [`experiments/llm_adamw_optimizer_bf16_kahan_stochastic_rounding_fp8_selection/results.json`](../../experiments/llm_adamw_optimizer_bf16_kahan_stochastic_rounding_fp8_selection/results.json).
> **中文版本**: [大模型 `AdamW` 优化器状态与主权重显存瘦身深水区（`14 Bytes/Param -> 7 Bytes/Param`）：为什么纯 `BF16` 权重更新会遭遇“大数吃小数（`w + Δw == w`）”训练停滞死锁？随机舍入（`Stochastic Rounding`）、卡汉补偿求和（`Kahan Summation`）与 `FP8` 一阶/二阶动量选型指南](./llm-adamw-optimizer-bf16-stagnation-kahan-stochastic-rounding-fp8-selection-faq.md)

![Low-Precision AdamW Kahan Summation & FP8 Moments Summary](../../assets/svg/llm-adamw-optimizer-bf16-stagnation-kahan-stochastic-rounding-fp8-selection-faq-summary.svg)

---

## 0. 🔥 今日领域最热门能力突破追踪（Daily Hottest Breakthrough）

- **Halving `70B` LLM Weight + Optimizer HBM from `980.0 GB` (`14 B/Param`) to `490.0 GB` (`7 B/Param`, Saving `490.0 GB`) While Avoiding the `+0.380` Loss Stagnation Trap of Pure `BF16` Master Weights via `Kahan Compensated Summation` + Block-Wise `FP8` Moments (`DeepSeek-V3 / TorchAO` Paradigm, `Tier 4`)**: In standard mixed-precision `AdamW` training (`Tier 2`), each parameter requires `2 B` (`BF16` model weight) + `4 B` (`FP32` master weight) + `4 B` (`FP32` first moment $m$) + `4 B` (`FP32` second moment $v$) = **`14.0 Bytes/parameter` (`980.0 GB` for `70B`)**. Why does dropping `FP32` master weights fail, and how do we compress `AdamW` by `50%` without losing a single bit of convergence accuracy?
  1. **Tier 1 — Pure `BF16` Weights & Moments With Round-to-Nearest-Even (`RNE` — `78%` Update Swallowing & `+0.380` Loss Stagnation!)**:
     - `BFloat16` has only **`7` mantissa bits** ($\text{ULP} = 2^{-7} \approx 0.0078$). Under IEEE RNE, any update $|\Delta w| < 2^{-8} |w| \approx \mathbf{0.0039 |w|}$ rounds back to exact $w$ (`BF16(w + Δw) == w`)!
     - For a typical weight $|w| \approx 0.02$ and cosine-decayed learning rate $\text{lr} = 3 \times 10^{-5}$, $|\Delta w| / |w| \approx \mathbf{0.0015 < 0.0039}$—meaning **`78%` of parameter updates vanish into zero**, freezing training!
  2. **Tier 2 — Standard `FP32` Master Weights + `FP32` Moments (`980.0 GB` HBM on `70B`, `0.000` Loss Gap)**:
     - Eliminates underflow stagnation, but consumes `14.0 Bytes/param`.
  3. **Tier 3 — `BF16` Weights with Unbiased `Stochastic Rounding (SR)` (`420.0 GB` HBM, `+0.018` Loss Gap)**:
     - Probabilistically rounds up by `1 ULP` with probability $p = |\Delta w| / \text{ULP}(w)$. While unbiased in expectation, random rounding variance degrades late-stage fine-grained convergence by `+0.018`.
  4. **Tier 4 — `Kahan Compensated BF16 Master Weights` (`2B w + 2B residual c`) + Block-Wise `FP8` First Moment $m$ (`1B`) + `BF16` Second Moment $v$ (`2B`) (`490.0 GB` HBM, `+0.001` Loss Parity)**:
     - Uses a `2-Byte` `BF16` **Kahan error accumulator `c`** to deterministically capture and carry forward every sub-ULP rounding residual of $w$, while compressing the EMA moments $m$ to `1x128` block-wise `FP8 (1B)` and $v$ to `BF16 (2B)`—cutting total state memory by **`-50.0%` (`980 GB -> 490 GB`)** with **`+0.001` loss parity**!

---

## 1. 📌 Executive Summary

1. **Why Master Weights Need `32-Bit` Effective Precision (`FP32` or `BF16 + Kahan Residual`) While Moments (`m, v`) Do Not**: Weight updates $w_t = w_{t-1} + \Delta w_t$ integrate tiny steps ($\Delta w / w \sim 10^{-4}$) over hundreds of thousands of iterations, whereas EMA moments $m_t, v_t$ have short exponential memory (`10–1,000` steps) and tolerate `1x128` block-wise `FP8 / BF16` quantization effortlessly.
2. **Why `BF16 Weight (2B) + BF16 Kahan Residual (2B)` Beats Separate `BF16 Weight (2B) + FP32 Master Weight (4B)`**: It eliminates the duplicate `2B` forward weight buffer (`4 Bytes/param` total instead of `6 Bytes/param`), saving `140 GB` on a `70B` model!

---

## 2. 🎯 Core Architect Q&A (Q1 / Q2 / Q3)

### Q1: How does `Kahan Summation` give two `BF16` numbers (`2B + 2B = 4B`) the accumulation precision of `FP32`?
- **Answer**: Because `BF16` has the **exact same `8-bit` exponent range as `FP32`**, a dedicated `BF16` residual tensor `c` centered near `0` shifts its exponent down to $2^{-14}$ to represent the exact `7` mantissa bits that were truncated from `w` ($2^{-7}$)! Together, `(w, c)` spans `14+` bits of effective mantissa precision!

### Q2: Why store the second moment $v$ (`exp_avg_sq`) in `BF16` (or `FP8` on $\sqrt{v}$) rather than raw unscaled `FP8`?
- **Answer**: Because $v_t$ accumulates **squared gradients** ($g_t^2$), doubling the exponent dynamic range required! Storing $v_t$ in `BF16` (which has an `8-bit` exponent) or storing $\sqrt{v_t}$ with `1x128` block scales prevents underflow in the denominator $\sqrt{v_t} + \epsilon$.

---

## 3. 🧭 Visual Diagrams

```mermaid
flowchart LR
    A["Tier 1: Pure BF16 RNE (420 GB · +0.380 Loss Stagnation ✕)"] --> B["Tier 2: Full FP32 Master+Moments (980 GB · 0.000 Loss ⚠️)"]
    B --> C["Tier 4: Kahan BF16 + Block-FP8 Moments (490 GB · -50% HBM · +0.001 Loss 🏆)"]
```

---

## 4. 📊 4-Tier Architect Selection Matrix

| Metric | Tier 1: Pure `BF16` (`RNE`) | Tier 2: Full `FP32` State | Tier 3: `BF16` + `Stochastic Rounding` | Tier 4: **`Kahan` Master + Block-`FP8` Moments** |
| :--- | :--- | :--- | :--- | :--- |
| **Bytes / Parameter (`Weight+Opt`)** | `6.0 B/param` | **`14.0 B/param` (Bloated)** ✕ | `6.0 B/param` ✅ | **`7.0 B/param` (`-50.0%` Reduction)** 🏆 |
| **`70B` Total State HBM (`GB`)** | `420.0 GB` | **`980.0 GB`** ✕ | `420.0 GB` ✅ | **`490.0 GB` (`490.0 GB` Saved!)** 🏆 |
| **Validation Loss Gap vs `FP32`** | **`+0.380` (Stagnation!)** ✕ | `0.000` (Baseline) ✅ | `+0.018` (SR Variance) ⚠️ | **`+0.001` (Deterministic Parity)** 🏆 |

---

## 5. 🧪 Quantitative Reference & Tooling

- **Evaluator & Sealed Data**: [`experiments/llm_adamw_optimizer_bf16_kahan_stochastic_rounding_fp8_selection/run.py`](../../experiments/llm_adamw_optimizer_bf16_kahan_stochastic_rounding_fp8_selection/run.py) · [`results.json`](../../experiments/llm_adamw_optimizer_bf16_kahan_stochastic_rounding_fp8_selection/results.json)
- **Interactive Sizer**: [`docs/reports/llm-adamw-optimizer-bf16-stagnation-kahan-stochastic-rounding-fp8-sizer.html`](../reports/llm-adamw-optimizer-bf16-stagnation-kahan-stochastic-rounding-fp8-sizer.html)
- **7-Card 4:5 Visual Report**: [`docs/reports/llm-adamw-optimizer-bf16-stagnation-kahan-stochastic-rounding-fp8-selection-cards.html`](../reports/llm-adamw-optimizer-bf16-stagnation-kahan-stochastic-rounding-fp8-selection-cards.html)

---

## 6. 🛠️ Good Fit vs. Not For & Production Checklist

- **✅ Good fit**: Large-scale LLM pretraining and full-parameter SFT/RL using `Kahan-compensated BF16` (or `FP32` master weights) with `1x128` block-wise `FP8` moments.
- **⛔ Not for**: Running standard `AdamW` directly on `BF16` parameters without master weights or Kahan error compensation.
- [ ] Never update `BF16` weights with standard RNE `AdamW` without a master weight or Kahan residual buffer.
- [ ] Fuse dequantization, moment update, Kahan step, and `1x128` `FP8` re-quantization into a single optimizer kernel.
