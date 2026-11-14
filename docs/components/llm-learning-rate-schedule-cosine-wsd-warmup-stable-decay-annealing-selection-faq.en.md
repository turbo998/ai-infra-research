# LLM Learning Rate Schedule (`Cosine` vs. `WSD` Warmup-Stable-Decay) & Continual Annealing Selection Guide: `Cosine` Re-Warmup Shock vs. `1/sqrt(t)` vs. Standard `WSD` vs. `WSD (1-sqrt) + High-Quality Annealing + EMA`

> **Language / 语言**: [中文](./llm-learning-rate-schedule-cosine-wsd-warmup-stable-decay-annealing-selection-faq.md) | **English (Current)**  
> **End-to-End Interactive Artifacts**: [🧮 Interactive WSD & Annealing Sizer](../reports/llm-learning-rate-schedule-cosine-wsd-warmup-stable-decay-annealing-sizer.html) · [🃏 7-Card 4:5 Visual Report](../reports/llm-learning-rate-schedule-cosine-wsd-warmup-stable-decay-annealing-selection-cards.html) · [🧪 Sealed Evaluator & Data](../../experiments/llm_lr_schedule_cosine_wsd_annealing_selection/run.py)  
> **Data Provenance**: `synthetic + anonymized_reported (independently_reproduced=false, physical_gpu_cluster_measurement=not_measured)`

![LLM Learning Rate Schedule (Cosine vs. WSD) & Continual Annealing Selection Summary](../../assets/svg/llm-learning-rate-schedule-cosine-wsd-warmup-stable-decay-annealing-selection-faq-summary.svg)

---

## 🔥 Daily Hottest Capability Breakthrough

- **Breakthrough Focus**: **Breaking the Fixed-Horizon Trap of `Cosine Annealing` in LLM Pretraining and Continual Pretraining—Why Re-Warming a Cosine-Decayed Checkpoint Wastes `1.2 Trillion` Tokens on "Re-Warmup Shock", and How `WSD (Warmup-Stable-Decay)` with `1 - sqrt` Cooldown, Mid-Training High-Quality Data Upsampling, and EMA Weight Averaging (`DeepSeek-V3 / Llama-3 / MiniCPM / OLMo-2`) Enables Zero-Waste Continual Scaling and Boosts Math/Coding Accuracy by `+10.6 pts` (`66.2 → 76.8`)**.
- **Why It Matters Today**: Enterprise foundation models are rarely trained once and frozen forever. When extending a `4T`-token model to `8T` tokens with newly curated code and domain data, a traditional `Cosine` schedule has already decayed its learning rate to `~0` at `4T`. Re-warming LR blasts the converged weights out of their loss basin, burning **`30%` (`1.2T` tokens)** of the new compute budget just to recover the prior validation loss!

---

## 📌 1. Executive Summary (3-Minute Architect Takeaways)

1. **Why `Cosine Decay` (`Tier 1`) Is a Multi-Million-Dollar Trap for Continual Pretraining**:
   - `Cosine` requires pre-specifying the exact total token horizon $T_{\max}$ on Day 1. Once LR reaches `0` at $T_{\max}$, re-warming LR for a second stage causes **Re-Warmup Shock**, wasting **`1,200B (1.2T) tokens`** on loss recovery ✕.
2. **How `WSD (Warmup-Stable-Decay)` (`Tier 3`) Decouples the "Infinite Trunk" from "On-Demand Release Branches"**:
   - **Stable Trunk (`85%–90%` of steps at constant peak LR $\eta_{\max}$)**: Never decays! Can be resumed and extended from `4T -> 8T -> 15T` with **`0` wasted recovery tokens**.
   - **Decay Branch (`Last 10%` of steps)**: Whenever a release milestone is needed, fork a branch from the `Stable` trunk and rapidly decay LR to `0`—triggering the **"WSD Cliff"** where validation loss drops below full-horizon `Cosine` (`1.78` vs. `1.84`) ✅!
3. **Production Gold Standard (`Tier 4`: `WSD` + `1 - sqrt(s)` Decay + `70%` High-Quality Annealing Data Mix + `EMA`)**:
   - During the final `10%` decay window, applying the steep-then-gentle **`1 - sqrt(s)`** decay schedule while upsampling verified math, code, and domain reasoning data to **`70%`** plus EMA weight averaging drives 8T validation loss to **`1.72`** and lifts math/coding benchmarks to **`76.8` (`+10.6 pts` over Cosine)** 🏆!

---

## 🎯 2. Architect & Customer Deep-Dive Q&A (Q1 / Q2 / Q3)

### Q1: Why does `WSD` validation loss look slightly higher than `Cosine` during the `Stable` phase before plunging during `Decay`?
- **Answer**: High constant learning rate maintains high stochastic gradient thermal energy (broad exploration across the parameter manifold). Once the `Decay` phase begins, thermal noise dissipates rapidly and the weights settle into the center of a flat minimum ("The WSD Cliff").

### Q2: How should we run continual pretraining if we only have a public model that has already decayed to `LR = 0`?
- **Answer**: Whenever available, start from the open-source model's **pre-cooldown (`Stable`) checkpoint** (e.g., OLMo-2 / MiniCPM stage-1 checkpoints). If only the final decayed checkpoint exists, cap peak LR to **`1/10–1/6` of pretraining peak LR (`3e-5`)**, use a `5%–10%` warmup, and mix `25%` replay data.

### Q3: Why is the `1 - sqrt(s)` decay curve superior to linear decay during the final `10%` annealing window?
- **Answer**: $\eta(s) = \eta_{\min} + (\eta_{\max} - \eta_{\min})(1 - \sqrt{s})$ drops steeply at $s \to 0$ to quickly quench high-LR oscillation, then spends more steps at low-to-moderate learning rates to absorb high-quality reasoning data.

---

## 🧭 3. Visual Architecture Diagrams (Mermaid)

```mermaid
flowchart TD
    A["LLM Pretraining & Continual Scaling LR Schedule Selection"] --> B{"Will the model be extended or branched across multiple token milestones?"}
    B -- "Yes, using fixed-horizon Cosine" --> T1["⛔ Tier 1: Cosine + Re-Warmup<br/>Re-warmup shock wastes 1.2T tokens recovering prior loss (Score 66.2) ✕"]
    B -- "Yes, using horizon-free schedule" --> C{"Does the schedule include a dedicated rapid cooldown phase?"}
    C -- "No (Continuous 1/sqrt(t))" --> T2["🟨 Tier 2: Inverse-Square-Root (1/sqrt(t))<br/>Never settles into sharp minimum, validation loss 1.88 (Score 63.5) ⚠️"]
    C -- "Yes (WSD Warmup-Stable-Decay)" --> D{"Co-annealing with high-purity data & EMA?"}
    D -- "Uniform data + linear decay" --> T3["🟦 Tier 3: Standard WSD (90% Stable + 10% Decay)<br/>0 wasted tokens, anytime branching, validation loss 1.78 (Score 71.0) ✅"]
    D -- "1-sqrt decay + 70% HQ Mix + EMA (Recommended)" --> T4["🏆 Tier 4: WSD (1-sqrt) + High-Quality Annealing + EMA<br/>0 wasted tokens, validation loss 1.72, Math/Code Score 76.8 (+10.6 pts)"]
```

---

## 📊 4. Architect Decision Matrix (4 Tiers Compared)

| Dimension | 🟥 Tier 1: `Cosine + Re-Warmup` | 🟨 Tier 2: `Inverse-Sqrt` | 🟦 Tier 3: Standard `WSD` | 🟩 Tier 4: `WSD (1-sqrt) + HQ Mix + EMA` (Recommended 🏆) |
| :--- | :--- | :--- | :--- | :--- |
| **Wasted Recovery Tokens (4T→8T)** | `1,200B (1.2T)` ✕ | **`0.0B`** ✅ | **`0.0B`** ✅ | **`0.0B` (Zero Waste)** 🏆 |
| **Final 8T Validation Loss** | `1.84` | `1.88` ✕ | `1.78` ✅ | **`1.72` (Lowest)** 🏆 |
| **Math & Coding Benchmark Score** | `66.2` | `63.5` | `71.0` ✅ | **`76.8` (`+10.6 pts`)** 🏆 |
| **Anytime Milestone Branching** | Unsupported ✕ | Supported | **100% Native** ✅ | **Infinite Trunk + HQ Branches** 🏆 |

---

## 🧪 5. Quantitative Case Study & 7-Card 4:5 Benchmark Report

```chart
{
  "data": {
    "values": [
      { "tier": "Tier 1: Cosine + Re-Warmup (1.2T Wasted)", "math_code_score": 66.2 },
      { "tier": "Tier 2: Inverse-Sqrt (No Cooldown)", "math_code_score": 63.5 },
      { "tier": "Tier 3: Standard WSD (Uniform Data)", "math_code_score": 71.0 },
      { "tier": "Tier 4: WSD (1-sqrt) + HQ Annealing + EMA", "math_code_score": 76.8 }
    ]
  },
  "semantic_types": {
    "tier": "Category",
    "math_code_score": "Score"
  },
  "chart_spec": {
    "chartType": "Bar Chart",
    "title": "WSD + High-Quality Annealing boosts 8T math/code score by +10.6 pts with zero re-warmup waste",
    "subtitle": "Composite Math & Coding benchmark score after extending training from 4T to 8T tokens",
    "encodings": {
      "x": "tier",
      "y": "math_code_score"
    }
  }
}
```

- 🧮 **Interactive WSD & Annealing Sizer**: [`docs/reports/llm-learning-rate-schedule-cosine-wsd-warmup-stable-decay-annealing-sizer.html`](../reports/llm-learning-rate-schedule-cosine-wsd-warmup-stable-decay-annealing-sizer.html)
- 🃏 **7-Card 4:5 Visual Benchmark Report**: [`docs/reports/llm-learning-rate-schedule-cosine-wsd-warmup-stable-decay-annealing-selection-cards.html`](../reports/llm-learning-rate-schedule-cosine-wsd-warmup-stable-decay-annealing-selection-cards.html)

---

## 🛠️ 6. Production Verification Checklist

- [ ] **Archive Un-Decayed `Stable` Trunk Checkpoints + Optimizer States**: Always branch off `Decay` runs from the `Stable` checkpoint while preserving the `Stable` trunk for future scaling.
- [ ] **Use `1 - sqrt(s)` Decay Curve During Cooldown**: Rapidly quench high-LR noise before fine-settling into the minimum.
- [ ] **Upsample High-Purity Math/Code/Domain Data During `Last 10% Decay`**: Reserve scarce gold-standard synthetic and domain corpora for the annealing phase.
- [ ] **Maintain EMA Weight Averaging (`decay = 0.999`) Across Cooldown Checkpoints**: Smooth out final step-to-step variance.
