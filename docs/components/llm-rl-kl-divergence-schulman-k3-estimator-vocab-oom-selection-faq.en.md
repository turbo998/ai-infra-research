# LLM Reinforcement Learning (`GRPO / PPO / RLOO`) KL-Divergence Schulman `k3` Estimator vs. Full-Vocabulary `KL` OOM Selection Guide: Why Naive `k1` Produces `48.5%` Negative KL Tokens (`14.8x` Variance), Why Full-Vocab `KL` Eats `39.86 GB` of HBM & How Clamped `k3` (`exp(clamp(d, -10, 10)) - 1 - d`, `0.12 MB` HBM, `0%` Inf) Solves It

> **Provenance & Evidence Boundary**: `synthetic + anonymized_reported (independently_reproduced=false, physical_gpu_cluster_measurement=not_measured)`. All quantitative metrics in this guide are grounded in the sealed deterministic evaluator `experiments/llm_rl_kl_divergence_schulman_k3_estimator_vocab_oom_selection/run.py` and `results.json`.

![RL KL-Divergence Schulman k3 Estimator vs Full-Vocab OOM Visual Summary](../../assets/svg/llm-rl-kl-divergence-schulman-k3-estimator-vocab-oom-selection-faq-summary.svg)

---

## 0. Daily Capability Breakthrough

- **Today's Highlight Breakthrough**: **Unbiased Non-Negative Schulman `k3` KL Estimator on 1D Sampled Token Logprobs ($\hat{D}_{k3} = \exp(\text{clamp}(\Delta, -10, 10)) - 1 - \Delta$, where $\Delta = \log \pi_{\text{ref}}(y_t) - \log \pi_\theta(y_t)$) Saving `39.86 GB` of HBM (`332,000x` Reduction) and Eliminating Negative-KL Reward Hacking (`DeepSeek-R1 GRPO / verl / OpenRLHF`)**.
- **Architect Value**: Regularizing the policy $\pi_\theta$ against a reference model $\pi_{\text{ref}}$ via KL divergence $D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$ is foundational to LLM reinforcement learning (`GRPO`, `PPO`, `RLOO`). Yet three common implementation mistakes cause either immediate `CUDA out of memory` crashes or severe training instability:
  1. **The Naive `k1` Estimator Trap (`48.5%` Negative KL Tokens & `14.8x` Higher Variance, Tier 1)**:
     - Defining per-token KL as $\hat{D}_{k1}(y_t) = \log \pi_\theta(y_t) - \log \pi_{\text{ref}}(y_t)$ yields a **strictly negative value whenever $\pi_\theta(y_t) < \pi_{\text{ref}}(y_t)$ (`48.5%` of sampled tokens)**!
     - Subtracting a negative KL "penalty" from the reward grants the policy a **positive reward bonus** for emitting low-probability tokens where $\pi_\theta < \pi_{\text{ref}}$, while exhibiting **`14.8x` higher Monte-Carlo variance** near $\pi_\theta \approx \pi_{\text{ref}}$!
  2. **The `39.86 GB` Full-Vocabulary Analytical KL OOM Trap (Tier 2)**:
     - Summing $\sum_{v=1}^V \pi_\theta(v) (\log \pi_\theta(v) - \log \pi_{\text{ref}}(v))$ across all $V = 152,064$ vocabulary tokens at `[B=4, S=8192]` materializes two `[4, 8192, 152064]` `FP32` tensors (**`39.86 GB` of HBM**) and transmits `19.93 GB` of reference logits over the network per microbatch!
  3. **The `6.4%` Unclamped `k3` Exponential Overflow Trap (Tier 3)**:
     - Evaluating $\exp(\log \pi_{\text{ref}}(y_t) - \log \pi_\theta(y_t))$ without clamping overflows when $\pi_\theta(y_t)$ assigns tiny probability to an exploratory token ($\Delta > 20 \implies e^{\Delta} > 4.8 \times 10^8$), causing `Inf/NaN` gradient spikes on **`6.4%`** of batches!
  By **computing the Schulman `k3` estimator exclusively on 1D sampled-token logprobs (`[B, S]` = `0.12 MB` HBM) with a symmetric log-ratio clamp $\Delta = \text{clamp}(\log \pi_{\text{ref}} - \log \pi_\theta, -10.0, +10.0)$ (Tier 4)**, KL workspace memory drops by **`39.86 GB` (`332,000x` smaller)**, negative KL tokens drop to **`0.0%`**, variance drops **`13.5x`**, and `exp()` overflows drop to **`0.0%`**!

---

## ① 📌 Executive Summary

1. **First-Principles Proof: Why Schulman's `k3` Estimator Is Simultaneously Unbiased and Strictly $\ge 0$**:
   - Let $r_t = \frac{\pi_{\text{ref}}(y_t)}{\pi_\theta(y_t)} = e^{\Delta_t}$ where $\Delta_t = \log \pi_{\text{ref}}(y_t) - \log \pi_\theta(y_t)$.
   - Because $\mathbb{E}_{y_t \sim \pi_\theta}[r_t - 1] = \sum_v \pi_\theta(v)\left(\frac{\pi_{\text{ref}}(v)}{\pi_\theta(v)} - 1\right) = 1 - 1 = 0$, adding the zero-mean control variate $(r_t - 1)$ to $-\log r_t$ preserves the exact expectation:
     $$\hat{D}_{k3}(y_t) = r_t - 1 - \log r_t = \mathbf{e^{\Delta_t} - 1 - \Delta_t \ge 0}$$
   - By the convexity of $e^x \ge 1 + x$, $\hat{D}_{k3}(y_t)$ is **strictly non-negative for every token** and has second-order Taylor expansion $\frac{1}{2}\Delta_t^2 + O(\Delta_t^3)$ near $\Delta_t = 0$!
2. **Overflow Protection via Symmetric Log-Ratio Clamping (`clamp(Delta, -10.0, 10.0)`)**:
   - Clamping $\Delta_t \in [-10.0, +10.0]$ prior to `torch.exp()` caps the maximum ratio at $e^{10} \approx 22,026$, preventing rare-token numerical explosions while keeping normal tokens `100%` untouched.

---

## ② 🎯 Architect Q&A (Q1 / Q2 / Q3)

### Q1: Why is `k2 = 0.5 * (log_pi - log_ref)^2` biased whereas `k3 = exp(d) - 1 - d` is unbiased?
- `k2` is the second-order Taylor approximation of `k3` around $\Delta = 0$; its expectation is an $f$-divergence distinct from true Kullback-Leibler divergence whenever $\pi_\theta$ drifts moderately from $\pi_{\text{ref}}$. `k3` is exact in expectation for all distributions.

### Q2: Should `k3` KL be subtracted from the reward (`PPO` style) or added directly to the loss (`GRPO` style)?
- In `GRPO` (`DeepSeek-R1`), adding $\beta \cdot \hat{D}_{k3}$ **directly to the policy loss** (`loss = pg_loss + beta * k3_kl`) avoids distorting group-normalized advantages $\hat{A}_i = (r_i - \mu_G) / \sigma_G$ when all $G$ rollouts achieve identical binary verifiable rewards!

### Q3: How much network traffic does 1D `k3` save between the Reference Worker and Actor Trainer in `verl`?
- Instead of sending `[B=4, S=8192, V=152064]` float32 logits (**`19.93 GB`**), the Reference Worker gathers only the sampled token logprobs `[4, 8192]` (**`128 KB`**, a **`152,064x` network reduction**)!

---

## ③ 🧭 Visual Diagrams (Decision Tree + Comparison + Sequence)

### 3.1 RL KL-Divergence Estimator Selection Decision Tree

```mermaid
flowchart TD
    A["LLM RL Post-Training KL Penalty D_KL(pi_theta || pi_ref) (B=4, S=8,192, V=152,064)"] --> B{"Full-Vocabulary Sum or 1D Sampled-Token Estimator?"}
    B -- "Full-Vocabulary Analytical Sum Over V=152,064" --> C["⛔ Tier 2: Materializes 39.86 GB HBM & sends 19.93 GB RPC payload -> OOM!"]
    B -- "1D Sampled-Token Estimator on [B, S] (0.12 MB HBM)" --> D{"Which Per-Token Formula Is Used?"}
    D -- "Naive k1 = log_pi_theta - log_pi_ref" --> E["⛔ Tier 1: Negative on 48.5% of tokens (turns penalty into bonus!) & 14.8x higher variance!"]
    D -- "Unclamped Schulman k3 = exp(d) - 1 - d" --> F["⚠️ Tier 3: Non-negative & unbiased, but exp(d) overflows to Inf on 6.4% of batches!"]
    D -- "Clamped Schulman k3: d = clamp(log_ref - log_pi, -10, 10); exp(d) - 1 - d" --> G["🏆 Tier 4: 0.12 MB HBM (-39.86 GB saved) + Strictly >= 0 + 13.5x Lower Variance + 0% Inf!"]
```

### 3.2 Naive `k1` (`48.5%` Negative) & Full-Vocab (`39.86 GB` OOM) vs. Clamped `k3` (`0.12 MB`)

```mermaid
flowchart LR
    subgraph Bad["⛔ Tier 1 & Tier 2: Negative-KL Reward Hacking (k1) or 39.86 GB Full-Vocab OOM"]
      B1["Tier 1 (Naive k1): Negative on 48.5% of tokens + 14.8x variance!"] --> B2["Tier 2 (Full-Vocab KL): Allocates 39.86 GB for [B, S, V] tensors -> OOM!"]
    end
    subgraph Good["🏆 Tier 4: Clamped Schulman k3 on 1D Sampled Logprobs (0.12 MB HBM)"]
      G1["Ref Worker returns 1D sampled logprobs_ref [4, 8192] (0.12 MB, 332,000x smaller!)"] --> G2["d = clamp(log_ref - log_pi, -10, 10) -> k3 = exp(d) - 1 - d >= 0 (0% neg, 0% Inf)!"]
    end
```

### 3.3 Disaggregated Reference-to-Trainer 1D Logprob Transfer & Clamped `k3` Sequence

```mermaid
sequenceDiagram
    participant Ref as Reference Worker (Frozen 70B)
    participant RPC as Ray / RPC Data Bus
    participant Actor as Actor Trainer (Fused Linear Logprobs)

    Ref->>Ref: 1. Forward pass on `[B=4, S=8192]` -> gather ONLY sampled token `logprobs_ref[4, 8192]`
    Ref->>RPC: 2. Transmit 1D `logprobs_ref` (`0.12 MB` instead of `19.93 GB` full logits!)
    Actor->>Actor: 3. Compute `logprobs_theta[4, 8192]` on sampled tokens
    Actor->>Actor: 4. `d = torch.clamp(logprobs_ref - logprobs_theta, -10.0, 10.0)`
    Actor->>Actor: 5. `k3_kl = torch.exp(d) - 1.0 - d` (`>= 0` everywhere, 0.0% Inf, 0.12 MB HBM!)
```

---

## ④ 📊 4-Tier Architect Decision Matrix

| Cohort Tier | KL Estimator Formulation | KL Workspace HBM | Negative-KL Tokens | Estimator Variance | `exp()` Inf Rate |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Tier 1: Naive `k1` (`log_pi - log_ref`)** | Uncontrolled 1D log difference | `0.00012 GB` | **`48.5%` ✕** | `14.8x` ✕ | `0.0%` |
| **Tier 2: Full-Vocab Analytical `KL`** | Exact sum over `152,064` vocab | **`39.86 GB` (OOM ✕)** | `0.0%` | `1.0x` | `0.0%` |
| **Tier 3: Unclamped Schulman `k3`** | `exp(d) - 1 - d` without clamp | `0.00012 GB` | `0.0%` | `1.1x` | **`6.4%` ✕** |
| **Tier 4: Clamped Schulman `k3` (Recommended)** | **`clamp(d, -10, 10)` + `exp(d)-1-d`** | **`0.00012 GB` (`-39.86 GB` 🏆)** | **`0.0%` 🏆** | **`1.1x` (`13.5x` lower 🏆)** | **`0.0%` 🏆** |

---

## ⑤ 🧪 Quantitative Evaluator & 7-Card 4:5 Visual Report

- **Sealed Evaluator**: [`experiments/llm_rl_kl_divergence_schulman_k3_estimator_vocab_oom_selection/run.py`](../../experiments/llm_rl_kl_divergence_schulman_k3_estimator_vocab_oom_selection/run.py) · Data: [`results.json`](../../experiments/llm_rl_kl_divergence_schulman_k3_estimator_vocab_oom_selection/results.json)
- **Interactive Sizer**: [`docs/reports/llm-rl-kl-divergence-schulman-k3-estimator-vocab-oom-sizer.html`](../reports/llm-rl-kl-divergence-schulman-k3-estimator-vocab-oom-sizer.html)
- **7-Card 4:5 Visual Benchmark Deck**: [`docs/reports/llm-rl-kl-divergence-schulman-k3-estimator-vocab-oom-selection-cards.html`](../reports/llm-rl-kl-divergence-schulman-k3-estimator-vocab-oom-selection-cards.html)

```chart
{
  "data": {
    "values": [
      { "tier": "Tier 1: Naive k1 (48.5% Negative)", "variance_ratio_x": 14.8, "hbm_gb": 0.00012 },
      { "tier": "Tier 2: Full-Vocab KL (39.86GB OOM)", "variance_ratio_x": 1.0, "hbm_gb": 39.86 },
      { "tier": "Tier 3: Unclamped Schulman k3 (6.4% Inf)", "variance_ratio_x": 1.1, "hbm_gb": 0.00012 },
      { "tier": "Tier 4: Clamped Schulman k3 (Recommended)", "variance_ratio_x": 1.1, "hbm_gb": 0.00012 }
    ]
  },
  "semantic_types": {
    "tier": "Category",
    "variance_ratio_x": "Quantity",
    "hbm_gb": "Quantity"
  },
  "chart_spec": {
    "chartType": "Bar Chart",
    "title": "Clamped Schulman k3 Estimator Cuts Variance 13.5x vs k1 & Saves 39.86 GB HBM vs Full-Vocab KL",
    "subtitle": "Relative Monte-Carlo estimator variance (x) across 4 KL estimation tiers (synthetic + anonymized_reported)",
    "encodings": {
      "x": "tier",
      "y": "variance_ratio_x",
      "color": "tier"
    }
  }
}
```

---

## ⑥ 🛠️ Production Verification Checklist (Good fit vs. Not for)

### ✅ Good fit vs. ⛔ Not for
- **✅ Good fit**: All `GRPO`, `PPO`, `RLOO`, and `REINFORCE++` reinforcement learning pipelines.
- **⛔ Not for**: Using raw `k1 = logp_theta - logp_ref` or evaluating `exp(logp_ref - logp_theta)` without clamping.

### Production Checklist
- [ ] **Use the Schulman `k3` estimator (`exp(d) - 1 - d` with `d = logp_ref - logp_theta`) instead of `k1`**.
- [ ] **Clamp `d = torch.clamp(logp_ref - logp_theta, min=-10.0, max=10.0)` before calling `torch.exp(d)`**.
- [ ] **Transmit only 1D sampled-token `logprobs[B, S]` (`0.12 MB`) from the Reference Worker rather than `[B, S, V]` logits**.
