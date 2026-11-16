# LLM Preference Alignment (`DPO / IPO / KTO`) Reference-Model `2x` Weight OOM, Asymmetric Pair Padding & Precomputed `ref_logps` + Fused Linear DPO Selection Guide: Why 70B DPO OOMs at `88.40 GB/GPU` on 8×H100-80GB & How Precomputed Reference Logprobs + Ragged `cu_seqlens` Packing + `Liger` Fused Linear DPO (`51.20 GB/GPU`, `-37.20 GB` Saved, `2.45x` Faster) Solves It

> **Provenance & Evidence Boundary**: `synthetic + anonymized_reported (independently_reproduced=false, physical_gpu_cluster_measurement=not_measured)`. All quantitative metrics in this guide are grounded in the sealed deterministic evaluator `experiments/llm_dpo_post_training_reference_model_oom_precomputed_logprobs_fused_loss_selection/run.py` and `results.json`.

![DPO Reference-Model OOM & Fused Linear Loss Visual Summary](../../assets/svg/llm-dpo-post-training-reference-model-oom-precomputed-logprobs-fused-loss-selection-faq-summary.svg)

---

## 0. Daily Capability Breakthrough

- **Today's Highlight Breakthrough**: **Offline Precomputed Reference Logprobs (`precompute_ref_log_probs` / `8 Bytes` per Pair) + Ragged 1D `cu_seqlens` `(chosen, rejected)` Pair Packing + SRAM-Tiled `LigerFusedLinearDPOLoss` (`TRL / Liger-Kernel / OpenRLHF / verl`)**.
- **Architect Value**: Why does a single 8×H100-80GB node easily train `70B` Full-Parameter SFT at `52 GB/GPU`, yet immediately crash with **`CUDA out of memory` (`88.40 GB/GPU` peak HBM)** and crawl at **`1,820 tokens/s/GPU`** when switched to standard **Direct Preference Optimization (`DPO`)**?
  Auditing naive DPO training pipelines (Tier 1) exposes three massive memory and compute bottlenecks:
  1. **The `17.50 GB/GPU` Frozen Reference Model ($\pi_{\text{ref}}$) Weight & `2x` Forward Compute Tax (Tier 1)**:
     - The DPO objective compares the trainable policy $\pi_\theta$ against a frozen reference model $\pi_{\text{ref}}$:
       $$\mathcal{L}_{\text{DPO}} = -\log \sigma\Bigl(\beta \bigl[(\log \pi_\theta(y_w|x) - \log \pi_{\text{ref}}(y_w|x)) - (\log \pi_\theta(y_l|x) - \log \pi_{\text{ref}}(y_l|x))\bigr]\Bigr)$$
     - Because $\pi_{\text{ref}}$ is **100% frozen** and the offline dataset $(x, y_w, y_l)$ is fixed, $\log \pi_{\text{ref}}(y_w|x)$ and $\log \pi_{\text{ref}}(y_l|x)$ are **just two constant `float32` scalars (`8 bytes` per preference pair!)**.
     - Yet naive trainers keep a full second `140 GB` copy of $\pi_{\text{ref}}$ in GPU HBM (`17.50 GB/GPU` across 8 GPUs) and recompute those two constant numbers on every epoch—wasting **`17.50 GB/GPU` of HBM** and **`33%` of total FLOPs**!
  2. **The `43.8%` Padding Compute Waste from `concat([chosen, rejected], dim=0)` (Tier 1 & Tier 2)**:
     - Because `chosen` (e.g., a concise `512`-token solution) and `rejected` (e.g., a rambling `4,096`-token failure) have asymmetric lengths, concatenating them along the batch dimension into `[2B, 4096]` pads the `512`-token sequence with `3,584` `<pad>` tokens—wasting **`43.8%`** of attention and MLP FLOPs!
  3. **The `17.30–19.93 GB/GPU` Materialized `[2B, L_max, V=152064]` Logits Tensor**:
     - Materializing full-vocabulary `FP32` logits across both `chosen` and `rejected` sequences simultaneously pushes peak HBM to **`88.40 GB/GPU` (`OOM`)**!
  By **precomputing `(ref_chosen_logp, ref_rejected_logp)` once and unloading $\pi_{\text{ref}}$ prior to optimizer allocation (or toggling `with model.disable_adapter():` under LoRA), packing `chosen` and `rejected` into a 1D ragged `cu_seqlens` tensor (`0%` padding), and computing target logprobs + DPO gradients inside a chunked `FusedLinearDPOLoss` kernel (Tier 4)**, peak HBM drops from `88.40 GB` to **`51.20 GB/GPU` (`-37.20 GB` saved)** and throughput jumps **`2.45x` (`1,820 -> 4,460 tok/s/GPU`)**!

---

## ① 📌 Executive Summary

1. **First-Principles Rule: Replace `140 GB` of Frozen $\pi_{\text{ref}}$ HBM with `8 Bytes/Sample` of Precomputed Scalars**:
   - Run 1 inference pass over the preference dataset before initializing `AdamW` states, store `(ref_chosen_logp, ref_rejected_logp)` in the dataset columns, and free $\pi_{\text{ref}}$ completely—reclaiming `17.50 GB/GPU` of HBM and eliminating 2 forward passes per pair!
   - When training LoRA/QLoRA DPO, $\pi_{\text{ref}}$ is simply the base model with adapters disabled (`with model.disable_adapter():`), requiring **zero extra weights** and zero offline storage.
2. **Ragged `cu_seqlens` Pair Packing + `FusedLinearDPOLoss`**:
   - Packing `chosen` ($L_w = 512$) and `rejected` ($L_l = 4,096$) into a 1D `[1, 4608]` ragged sequence (`cu_seqlens = [0, 512, 4608]`) eliminates `43.8%` padding waste, while tiling `lm_head` in SRAM saves another `17.30 GB/GPU` of materialized logits!

---

## ② 🎯 Architect Q&A (Q1 / Q2 / Q3)

### Q1: Why must $\pi_{\text{ref}}$ be unloaded *before* calling `optimizer = AdamW(policy.parameters())` (or before the first backward step)?
- Because PyTorch's FSDP / caching allocator reserves high-water-mark blocks when `AdamW` states (`m`, `v`, `master_weights`) are materialized. Unloading $\pi_{\text{ref}}$ and calling `torch.cuda.empty_cache()` prior to training ensures the `17.50 GB/GPU` is available for policy gradients and activations.

### Q2: How does `FusedLinearDPOLoss` compute the exact DPO gradient without storing `[N, V]` logits?
- In Pass 1, it streams micro-tiles of hidden states $\mathbf{h}_t$ through $\mathbf{W}_{\text{lm\_head}}^T$ in SRAM to compute online log-sum-exp $\text{LSE}_t$ and target token log-probabilities $\log p_\theta(y_t \mid x)$, summing them into sequence scalars $\log \pi_\theta(y_w\mid x)$ and $\log \pi_\theta(y_l\mid x)$. Once the scalar DPO sigmoid coefficient $w = -\beta \sigma(-\beta \Delta)$ is known, it computes chunked gradients $d\mathbf{H}$ and $d\mathbf{W}$ in-place!

### Q3: Can we also share the prompt prefix $x$ between `chosen` ($x + y_w$) and `rejected` ($x + y_l$)?
- Yes! When prompt $x$ is long relative to responses $y_w, y_l$, tree/cascade attention over the shared prefix $[x, y_w, y_l]$ computes the prompt KV states only once instead of twice.

---

## ③ 🧭 Visual Diagrams (Decision Tree + Comparison + Sequence)

### 3.1 DPO Post-Training Memory & Throughput Optimization Decision Tree

```mermaid
flowchart TD
    A["70B Full-Parameter DPO / IPO / KTO Training (8x H100-80GB, V=152,064, L_max=8,192)"] --> B{"How Are Reference Model (pi_ref) Logprobs Computed?"}
    B -- "Keep frozen 70B pi_ref in HBM & recompute every step" --> C["⛔ Tier 1: Wastes 17.50 GB/GPU HBM & 2 forward passes/pair -> 88.40 GB/GPU OOM!"]
    B -- "Precompute (ref_chosen_logp, ref_rejected_logp) offline & unload pi_ref" --> D{"How Are (chosen, rejected) Pairs Batched?"}
    D -- "Concatenate [2B, L_max] with right-padding" --> E["⚠️ Tier 2: Fits in 70.90 GB, but asymmetric lengths waste 43.8% of FLOPs on <pad> (2,640 tok/s)!"]
    D -- "Pack into 1D Ragged Sequence with cu_seqlens (FlashAttention varlen)" --> F{"Fuse lm_head Projection + DPO Loss?"}
    F -- "No: Materialize [N, 152064] FP32 Logits" --> G["⚠️ Tier 3: 3,790 tok/s, but materialized logits still consume 17.30 GB/GPU (68.50 GB peak)!"]
    F -- "Yes: Liger FusedLinearDPOLoss (SRAM-tiled)" --> H["🏆 Tier 4: 51.20 GB/GPU Peak HBM (-37.20 GB saved) + 4,460 tok/s/GPU (2.45x faster)!"]
```

### 3.2 Dual-Model Padded DPO (`88.40 GB` OOM) vs. Precomputed + Ragged + Fused DPO (`51.20 GB`)

```mermaid
flowchart LR
    subgraph Bad["⛔ Tier 1: Dual 70B Models in HBM + Padded [2B, L_max] + Materialized Logits (88.40 GB OOM!)"]
      B1["Frozen pi_ref eats 17.50 GB/GPU + 2 extra forwards/pair"] --> B2["512-tok chosen padded to 4,096 (43.8% pad waste)"]
      B2 --> B3["[N, 152064] logits eat 19.7 GB -> 88.40 GB/GPU OOM!"]
    end
    subgraph Good["🏆 Tier 4: Precomputed 8B Ref Scalars + Ragged cu_seqlens + FusedLinearDPOLoss (51.20 GB)"]
      G1["Precompute ref_logps once & unload pi_ref (-17.50 GB)"] --> G2["Pack chosen(512)+rejected(4096) into 1D 4,608-tok array (0% pad)"]
      G2 --> G3["SRAM-tiled FusedLinearDPOLoss (-17.30 GB) -> 51.20 GB/GPU, 2.45x faster!"]
    end
```

### 3.3 Offline `ref_logps` Precomputation, Ragged `cu_seqlens` Forward & Fused DPO Backward Sequence

```mermaid
sequenceDiagram
    participant Pre as Precompute Pass (Run Once)
    participant Data as Dataset (ref_chosen_logp, ref_rejected_logp)
    participant Policy as 70B Policy Backbone (FlashAttn varlen)
    participant Fused as Liger FusedLinearDPOLoss (SRAM Tiled)

    Pre->>Data: 1. Compute 2 float32 scalars `(ref_w, ref_l)` per pair once & free `pi_ref` (-17.5 GB/GPU!)
    loop DPO Training Step
        Data->>Policy: 2. Pack `chosen (512)` + `rejected (4096)` into 1D `input_ids[4608]`, `cu_seqlens=[0, 512, 4608]`
        Policy->>Fused: 3. Emit `hidden_states[4608, D]` (0% padding FLOPs wasted!)
        Fused->>Fused: 4. Tile `lm_head` in SRAM: accumulate `log pi_theta(y_w)` & `log pi_theta(y_l)` via online LSE
        Fused-->>Policy: 5. Compute DPO sigmoid scaling & emit fused `dH` + `dW_lm_head` in-place (51.20 GB peak HBM!)
    end
```

---

## ④ 📊 4-Tier Architect Decision Matrix

| Cohort Tier | 70B DPO Training Architecture | Peak HBM / GPU (`8x H100`) | Throughput (`tok/s/GPU`) | Padding FLOP Waste | 80GB H100 Fit |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Tier 1: Dual Model + Padded `[2B, L]` + Logits** | Keeps $\pi_{\text{ref}}$ + pads + `[N, V]` logits | **`88.40 GB` (OOM ✕)** | `1,820 tok/s` ✕ | `43.8%` ✕ | ⛔ **OOM Crash** |
| **Tier 2: Precomputed Ref (Padded + Logits)** | Unloads $\pi_{\text{ref}}$; still pads & materializes | `70.90 GB` | `2,640 tok/s` | `43.8%` ✕ | ⚠️ **Tight / Slow** |
| **Tier 3: Precomputed Ref + Ragged `cu_seqlens`** | Unpadded 1D pack, unfused `[N, V]` logits | `68.50 GB` | `3,790 tok/s` | `0.0%` 🏆 | ✅ **Fits (`8K`)** |
| **Tier 4: Precomputed + Ragged + Fused Loss (Recommended)** | **Precomputed ref + 1D pack + fused DPO** | **`51.20 GB` (`-37.2 GB` 🏆)** | **`4,460 tok/s` (`2.45x` 🏆)** | **`0.0%` 🏆** | 🏆 **Fits `16K+` Easily** |

---

## ⑤ 🧪 Quantitative Evaluator & 7-Card 4:5 Visual Report

- **Sealed Evaluator**: [`experiments/llm_dpo_post_training_reference_model_oom_precomputed_logprobs_fused_loss_selection/run.py`](../../experiments/llm_dpo_post_training_reference_model_oom_precomputed_logprobs_fused_loss_selection/run.py) · Data: [`results.json`](../../experiments/llm_dpo_post_training_reference_model_oom_precomputed_logprobs_fused_loss_selection/results.json)
- **Interactive Sizer**: [`docs/reports/llm-dpo-post-training-reference-model-oom-precomputed-logprobs-fused-loss-sizer.html`](../reports/llm-dpo-post-training-reference-model-oom-precomputed-logprobs-fused-loss-sizer.html)
- **7-Card 4:5 Visual Benchmark Deck**: [`docs/reports/llm-dpo-post-training-reference-model-oom-precomputed-logprobs-fused-loss-selection-cards.html`](../reports/llm-dpo-post-training-reference-model-oom-precomputed-logprobs-fused-loss-selection-cards.html)

```chart
{
  "data": {
    "values": [
      { "tier": "Tier 1: Dual Model + Padded [2B, L]", "peak_hbm_gb": 88.40, "tps_per_gpu": 1820 },
      { "tier": "Tier 2: Precomputed Ref (Padded)", "peak_hbm_gb": 70.90, "tps_per_gpu": 2640 },
      { "tier": "Tier 3: Precomputed + Ragged cu_seqlens", "peak_hbm_gb": 68.50, "tps_per_gpu": 3790 },
      { "tier": "Tier 4: Precomputed + Pack + Fused DPO Loss", "peak_hbm_gb": 51.20, "tps_per_gpu": 4460 }
    ]
  },
  "semantic_types": {
    "tier": "Category",
    "peak_hbm_gb": "Quantity",
    "tps_per_gpu": "Quantity"
  },
  "chart_spec": {
    "chartType": "Bar Chart",
    "title": "Precomputed Ref Logprobs + Ragged Packing + Fused Loss Saves 37.2 GB/GPU & Gains 2.45x Throughput",
    "subtitle": "Peak per-GPU HBM usage (GB) on 8x H100-80GB for 70B DPO across 4 tiers (synthetic + anonymized_reported)",
    "encodings": {
      "x": "tier",
      "y": "peak_hbm_gb",
      "color": "tier"
    }
  }
}
```

---

## ⑥ 🛠️ Production Verification Checklist (Good fit vs. Not for)

### ✅ Good fit vs. ⛔ Not for
- **✅ Good fit**: All full-parameter and LoRA `DPO / IPO / KTO` preference alignment pipelines on large-vocabulary (`128K–256K`) or long-context (`8K–32K`) models.
- **⛔ Not for**: Keeping a frozen full-weight $\pi_{\text{ref}}$ resident in GPU HBM during multi-epoch DPO or right-padding asymmetric `(chosen, rejected)` pairs.

### Production Checklist
- [ ] **Precompute `(ref_chosen_logps, ref_rejected_logps)` once and unload $\pi_{\text{ref}}$ prior to training (or use `with model.disable_adapter():` for LoRA DPO)**.
- [ ] **Pack `chosen` and `rejected` sequences into 1D ragged batches (`cu_seqlens`) with `FlashAttention varlen`**.
- [ ] **Use `LigerFusedLinearDPOLoss` to avoid materializing `[2B, L, V]` logits in HBM**.
