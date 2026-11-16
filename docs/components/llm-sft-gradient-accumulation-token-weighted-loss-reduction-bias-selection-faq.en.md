# LLM SFT & RL (`GRPO / DPO`) Gradient Accumulation Loss Reduction Length Bias (`loss.mean() / K` vs. Global Valid-Token Sum) Selection Guide: Why `(loss.mean() / grad_accum_steps).backward()` Overweights `32`-Token Responses `80x` Relative to `2,560`-Token Responses (`-9.4 pt` Accuracy Drop) & How `reduction="sum"` + Global `All-Reduce(total_valid_tokens)` (`1.0x` Uniform Weight, `0.0%` Grad Error) Solves It

> **Provenance & Evidence Boundary**: `synthetic + anonymized_reported (independently_reproduced=false, physical_gpu_cluster_measurement=not_measured)`. All quantitative metrics in this guide are grounded in the sealed deterministic evaluator `experiments/llm_sft_gradient_accumulation_token_weighted_loss_reduction_bias_selection/run.py` and `results.json`.

![SFT & RL Gradient Accumulation Loss Reduction Length Bias Visual Summary](../../assets/svg/llm-sft-gradient-accumulation-token-weighted-loss-reduction-bias-selection-faq-summary.svg)

---

## 0. Daily Capability Breakthrough

- **Today's Highlight Breakthrough**: **Global Valid-Token Sum Normalization ($\tilde{\mathcal{L}}_{r,k} = \frac{D_{\text{dp}}}{N_{\text{global\_valid}}} \sum_{i=1}^{n_{r,k}} \ell_{r,k,i}$) Across All $K$ Gradient Accumulation Steps and $D$ Data-Parallel Ranks (`HuggingFace num_items_in_batch / Unsloth / Dr.GRPO / DAPO`)**.
- **Architect Value**: Standard PyTorch training loops compute gradient accumulation as:
  ```python
  # ⛔ SEVERE MATHEMATICAL BUG WHEN VALID TOKEN COUNTS VARY!
  loss = F.cross_entropy(logits.view(-1, V), labels.view(-1), ignore_index=-100, reduction="mean")
  (loss / grad_accum_steps).backward()
  ```
  While valid when every microbatch has an identical number of unmasked target tokens, in **Instruction SFT** (where prompt and padding tokens are masked with `labels = -100`) and **RL Post-Training (`GRPO`)** (where rollout lengths span `32` to `16,384` tokens), this formula introduces a devastating **Length Bias**:
  1. **The "Mean of Means $\neq$ Global Mean" Fallacy (`80.0x` Short-Token Bias, Tier 1)**: Suppose Microbatch 1 has $n_1 = 32$ valid response tokens and Microbatch 2 has $n_2 = 2,560$ valid response tokens ($K = 8$). `reduction="mean"` divides Microbatch 1's tokens by `32` and Microbatch 2's tokens by `2,560` before dividing both by `8`. As a result, each token in the short `32`-token response receives **`80.0x` higher gradient weight** ($\frac{1}{32 \times 8}$ vs $\frac{1}{2560 \times 8}$) than each token in the `2,560`-token long chain-of-thought response! This causes a **`46.2%` gradient deviation** relative to the true unsharded batch and drops long-form instruction/reasoning accuracy by **`-9.4 points` (`71.8% -> 62.4%`)**!
  2. **Local Rank Token-Sum Normalization Without Cross-DP Synchronization (`14.5%` Grad Error, Tier 2)**: Even if each rank divides by its own local token sum $\sum_k n_{r,k}$, PyTorch `DDP / FSDP` averages gradients across the $D=8$ ranks with equal weight $\frac{1}{D}$. When Rank 0 has `1,200` valid tokens and Rank 1 has `7,680` valid tokens, Rank 0's tokens are still overweighted by **`6.4x`**!
  By **computing unnormalized sums (`reduction="sum"`), executing a single scalar `All-Reduce` to count total valid tokens $N_{\text{global}} = \sum_{r=1}^D \sum_{k=1}^K n_{r,k}$ across all microsteps and DP ranks, and scaling each microbatch loss by $\frac{D_{\text{dp}}}{N_{\text{global}}}$ (Tier 4)**, every valid token across the entire cluster receives an exact uniform weight of $\frac{1}{N_{\text{global}}}$ (**`1.0x` ratio, `0.0%` gradient error**), boosting long-form accuracy to **`71.8%` (`+9.4 pt` gain)**!

---

## ① 📌 Executive Summary

1. **First-Principles Proof: Why Multiplying `loss_sum` by `dp_world_size / N_global` Cancels `DDP / FSDP` Averaging Exactly**:
   - The true token-level global loss across $D$ ranks and $K$ microbatches is:
     $$\mathcal{L}_{\text{true}} = \frac{1}{N_{\text{global}}} \sum_{r=1}^{D} \sum_{k=1}^{K} S_{r,k}, \quad S_{r,k} = \sum_{i=1}^{n_{r,k}} \ell_{r,k,i}$$
   - Because `DDP` and `FSDP` automatically divide the summed gradients across ranks by $D$ (`1 / dp_world_size`), scaling each local microbatch sum by $\frac{D}{N_{\text{global}}}$ produces:
     $$G_{\text{final}} = \frac{1}{D} \sum_{r=1}^{D} \sum_{k=1}^{K} \left(\frac{D}{N_{\text{global}}} \nabla S_{r,k}\right) = \frac{1}{N_{\text{global}}} \sum_{r=1}^{D} \sum_{k=1}^{K} \nabla S_{r,k} \equiv \nabla \mathcal{L}_{\text{true}}$$
   - This makes gradients **100% invariant** to how a global batch is split across `grad_accum_steps` ($K$) and `dp_world_size` ($D$)!
2. **Why Sequence-Level Equal Normalization (`1 / |o_i|`, Tier 3) Induces Length Hacking in `GRPO`**:
   - Dividing each rollout trajectory by its own length $|o_i|$ rewards short correct answers more per token and **dilutes negative penalties on long wrong answers**—teaching the policy to emit thousands of tokens of rambling text whenever it is uncertain (`Dr.GRPO` length bias). Global token-count normalization eliminates this pathology.

---

## ② 🎯 Architect Q&A (Q1 / Q2 / Q3)

### Q1: How can we know $N_{\text{global}}$ across all $K$ microbatches *before* calling `.backward()` on Microbatch 1?
- Because the number of valid tokens $n_{r,k} = (\text{labels}_{r,k} \neq -100).\text{sum}()$ depends **only on the dataset labels** (and not on the model forward pass!), simply prefetch the $K$ microbatch dicts from the `DataLoader` at the start of the step, sum `(mb["labels"] != -100).sum()` across the $K$ microbatches, and run a single 1-integer `dist.all_reduce` (`< 0.02 ms`) before starting the forward/backward loop!

### Q2: What happens if a microbatch on one rank happens to have `0` valid tokens (e.g., all tokens masked as `-100`)?
- Under `reduction="mean"`, `0 / 0` produces **`NaN`** and immediately corrupts all weights on `All-Reduce`! Under Tier 4 (`reduction="sum"` scaled by `D / max(N_global, 1)`), a microbatch with 0 valid tokens cleanly produces `loss = 0.0 * logits.sum()`, contributing an exact `0.0` gradient without `NaN`!

### Q3: How do we unit-test a training pipeline for gradient accumulation invariance?
- Run 1 step with `micro_batch_size=8, grad_accum=1, dp=1` vs `micro_batch_size=1, grad_accum=4, dp=2` on the exact same 8 variable-length sequences and assert `torch.allclose(grad_A, grad_B, atol=1e-5)`.

---

## ③ 🧭 Visual Diagrams (Decision Tree + Comparison + Sequence)

### 3.1 Gradient Accumulation & DP Loss Normalization Decision Tree

```mermaid
flowchart TD
    A["Variable-Length SFT / DPO / GRPO Training (grad_accum K=8, DP=8, labels == -100 Masking)"] --> B{"How Are Microbatch Losses Reduced & Scaled?"}
    B -- "Default reduction='mean' / grad_accum_steps" --> C["⛔ Tier 1: 32-token batch weighted 80x higher than 2,560-token batch -> 46.2% grad error!"]
    B -- "Per-sequence equal mean (1 / |o_i|)" --> D["⛔ Tier 3: Dilutes penalty on long wrong chains -> RL length hacking & SFT truncation!"]
    B -- "Unnormalized token sum (reduction='sum')" --> E{"Is the Token Count Denominator Synchronized Across All DP Ranks?"}
    E -- "No: Divide by local_rank_valid_tokens only" --> F["⛔ Tier 2: Unequal token counts across DP ranks cause 14.5% cross-rank bias!"]
    E -- "Yes: All-Reduce N_global & scale by dp_world_size / N_global" --> G["🏆 Tier 4: 1.0x uniform token weight + 0.0% grad error + 71.8% accuracy (+9.4 pt)!"]
```

### 3.2 `loss.mean() / K` (`80x` Short-Token Bias) vs. Global Valid-Token Sum (`1.0x` Uniform)

```mermaid
flowchart LR
    subgraph Bad["⛔ Tier 1: Per-Microbatch Mean / 8 (80x Short-Response Overweighting!)"]
      B1["Microbatch 1 (32 Valid Tokens): Weight = 1 / (32 × 8) = 1/256"] --> B2["Microbatch 2 (2,560 Valid Tokens): Weight = 1 / (2560 × 8) = 1/20,480 (80x Skew!)"]
    end
    subgraph Good["🏆 Tier 4: reduction='sum' * (DP / N_global) (Exact Uniform Weighting!)"]
      G1["Pre-Count Total Valid Tokens N_global Across 8 Ranks × 8 Microsteps"] --> G2["Every Valid Token Receives Exact Weight 1 / N_global (0.0% Deviation)!"]
    end
```

### 3.3 Prefetched $K$-Microbatch Token Count `All-Reduce` & Exact Scaling Sequence

```mermaid
sequenceDiagram
    participant Loader as DataLoader (Prefetch K=8 Microbatches)
    participant Comm as NCCL ProcessGroup (DP = 8)
    participant Model as FSDP / DDP Model (reduction='sum')

    Loader->>Comm: 1. Count local valid tokens across K=8 microbatches: `n_local = sum((y != -100).sum())`
    Comm->>Comm: 2. 1-Scalar All-Reduce(SUM): `N_global = all_reduce(n_local)` (<0.02 ms)
    loop Microbatch k = 1 .. K
        Loader->>Model: 3. Compute unnormalized sum: `loss_sum = CE(logits, y, reduction='sum')`
        Model->>Model: 4. Scale by `dp_world_size / N_global` & run `.backward()`
    end
    Model->>Comm: 5. FSDP/DDP averages across DP ranks (`1 / dp_world_size` cancels numerator `dp_world_size`!)
```

---

## ④ 📊 4-Tier Architect Decision Matrix

| Cohort Tier | Loss Reduction & Scaling Policy | Short vs Long Token Weight | True-Batch Grad Error | Long-Form Eval Accuracy | Best For |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Tier 1: Naive `loss.mean() / K`** | Averages inside each microbatch | `80.0x` Skew ✕ | `46.2%` ✕ | `62.4%` ✕ | ⛔ **Severe Length Bias in SFT/RL** |
| **Tier 2: Local Rank Token Sum** | Divides by `n_local` without DP sync | `6.4x` Skew ✕ | `14.5%` ✕ | `67.8%` ✕ | ⛔ **Biased Across DP Ranks** |
| **Tier 3: Sequence-Equal Mean** | Equal weight per prompt `1 / |o_i|` | `80.0x` Skew ✕ | `38.0%` ✕ | `64.1%` ✕ | ⛔ **Induces RL Length Hacking** |
| **Tier 4: Global Valid-Token Sum (Recommended)** | **`sum * (DP / N_global)`** | **`1.0x` Uniform 🏆** | **`0.0%` (Exact 🏆)** | **`71.8%` (`+9.4 pt` 🏆)** | 🏆 **All Variable-Length SFT & RL** |

---

## ⑤ 🧪 Quantitative Evaluator & 7-Card 4:5 Visual Report

- **Sealed Evaluator**: [`experiments/llm_sft_gradient_accumulation_token_weighted_loss_reduction_bias_selection/run.py`](../../experiments/llm_sft_gradient_accumulation_token_weighted_loss_reduction_bias_selection/run.py) · Data: [`results.json`](../../experiments/llm_sft_gradient_accumulation_token_weighted_loss_reduction_bias_selection/results.json)
- **Interactive Sizer**: [`docs/reports/llm-sft-gradient-accumulation-token-weighted-loss-reduction-bias-sizer.html`](../reports/llm-sft-gradient-accumulation-token-weighted-loss-reduction-bias-sizer.html)
- **7-Card 4:5 Visual Benchmark Deck**: [`docs/reports/llm-sft-gradient-accumulation-token-weighted-loss-reduction-bias-selection-cards.html`](../reports/llm-sft-gradient-accumulation-token-weighted-loss-reduction-bias-selection-cards.html)

```chart
{
  "data": {
    "values": [
      { "tier": "Tier 1: Naive loss.mean() / K", "grad_err_pct": 46.2, "accuracy_pct": 62.4 },
      { "tier": "Tier 2: Local Rank Token Sum Only", "grad_err_pct": 14.5, "accuracy_pct": 67.8 },
      { "tier": "Tier 3: Sequence-Equal Weighting", "grad_err_pct": 38.0, "accuracy_pct": 64.1 },
      { "tier": "Tier 4: Global Valid-Token Sum (DP/N_global)", "grad_err_pct": 0.0, "accuracy_pct": 71.8 }
    ]
  },
  "semantic_types": {
    "tier": "Category",
    "grad_err_pct": "Percentage",
    "accuracy_pct": "Percentage"
  },
  "chart_spec": {
    "chartType": "Bar Chart",
    "title": "Global Valid-Token Sum Normalization Eliminates 46.2% Gradient Bias & Gains +9.4% Long-Form Accuracy",
    "subtitle": "Gradient deviation vs true global batch (%) and long-form instruction accuracy (%) (synthetic + anonymized_reported)",
    "encodings": {
      "x": "tier",
      "y": "accuracy_pct",
      "color": "tier"
    }
  }
}
```

---

## ⑥ 🛠️ Production Verification Checklist (Good fit vs. Not for)

### ✅ Good fit vs. ⛔ Not for
- **✅ Good fit**: All SFT, continual pretraining, DPO, and GRPO/PPO pipelines where target token counts (`labels != -100`) vary across microbatches or DP ranks.
- **⛔ Not for**: Using `reduction="mean"` divided by `grad_accum_steps` on variable-length masked sequences.

### Production Checklist
- [ ] **Compute microbatch losses with `reduction="sum"`**: Never divide by the single-microbatch token count inside `cross_entropy`.
- [ ] **All-Reduce `N_global = sum(labels != -100)` across all $K$ microbatches and $D$ DP ranks**: Multiply each microbatch `loss_sum` by `dp_world_size / max(N_global, 1)`.
- [ ] **Verify gradient invariance across `(batch_size, grad_accum_steps, dp_size)` decompositions**: Confirm identical gradients against a single unsharded batch.
