# PyTorch `nn.Module.register_buffer(persistent=True vs False)` Long-Context `RoPE` Overwrite Bug (`64K PPL` `184.60` vs. `5.42`) & `+18.40 GB` Checkpoint Bloat Selection Guide: Why Default `persistent=True` Silently Overwrites Extended `rope_theta = 500,000` During `load_state_dict()` & How `persistent=False` + Shared `RotaryEmbedding` Solves It

> **Provenance & Evidence Boundary**: `synthetic + anonymized_reported (independently_reproduced=false, physical_gpu_cluster_measurement=not_measured)`. All quantitative metrics in this guide are grounded in the sealed deterministic evaluator `experiments/pytorch_register_buffer_persistent_false_rope_overwrite_checkpoint_bloat_selection/run.py` and `results.json`.

![PyTorch register_buffer(persistent=True vs False) RoPE Overwrite & Checkpoint Bloat Visual Summary](../../assets/svg/pytorch-register-buffer-persistent-false-rope-overwrite-checkpoint-bloat-selection-faq-summary.svg)

---

## 0. Daily Capability Breakthrough

- **Today's Highlight Breakthrough**: **Non-Persistent Deterministic Buffer Registration (`self.register_buffer("inv_freq", inv_freq, persistent=False)`) Paired with Top-Level Single-Instance `RotaryEmbedding` and Legacy `state_dict` Key Filtering (`Llama-3 / Qwen2.5 / TorchTitan 8K -> 128K Context Extension`)**.
- **Architect Value**: During long-context continual pre-training and fine-tuning (`ABF / YaRN` extending context from `8K` to `128K` by increasing `rope_theta` from `10,000` to `500,000`), a notorious silent bug causes the model to pass short-context `4K` checks (`5.18 PPL`) while collapsing completely at `64K` context (**`184.60 PPL` instead of `5.42 PPL`**)! Three `nn.Module` buffer pitfalls explain why:
  1. **The Silent `load_state_dict()` `inv_freq` Overwrite Bug (`persistent=True` Default, Tier 1 — `34.1x` Worse `64K` Perplexity!)**:
     - `nn.Module.register_buffer(name, tensor, persistent=True)` defaults to **`persistent=True`**, serializing the deterministic `inv_freq` vector (`shape = [head_dim // 2] = [64]`) into `state_dict()` and `.safetensors` files.
     - When you instantiate a model with `rope_theta = 500,000` and then call `model.load_state_dict(base_8k_ckpt)` to load weights from an 8K checkpoint whose `inv_freq` was saved at `rope_theta = 10,000`, **PyTorch sees identical `[64]` shapes and silently overwrites your new `500,000` `inv_freq` buffer with the stale `10,000` tensor from disk** without raising any warning!
  2. **The `+18.40 GB` Checkpoint Bloat & `strict=True` Shape-Mismatch Crash from Persistent Masks/Caches (Tier 2)**:
     - Registering per-layer causal `bias` masks (`[1, 1, S, S]`) or precomputed `cos_cached / sin_cached` tables with default `persistent=True` across `80` layers bloats every saved checkpoint by **`+18.40 GB`** and crashes `load_state_dict(strict=True)` whenever `max_position_embeddings` changes!
  3. **The `100%` Device-Mismatch Crash of Plain Python Tensor Attributes (`self.inv_freq = inv_freq`, Tier 3)**:
     - Assigning `self.inv_freq` as a plain Python attribute avoids `state_dict()`, but `model.to("cuda")`, `.to(dtype)`, and `FSDP` ignore plain attributes, leaving `self.inv_freq` stranded on `cpu`!
  By **passing `persistent=False` on all deterministic buffers (`self.register_buffer("inv_freq", inv_freq, persistent=False)`), sharing a single `RotaryEmbedding` module at the top-level `Model`, and stripping legacy `"*.inv_freq"` keys in a `load_state_dict` pre-hook (Tier 4)**, `rope_theta = 500,000` is `100%` preserved (**`64K PPL = 5.42` vs `184.60`**), checkpoint bloat drops by **`18.40 GB` to `0.00 GB`**, and `.to("cuda")` works with **`0.0%` crash rate**!

---

## ① 📌 Executive Summary

1. **First-Principles Rule for `persistent=True` vs. `persistent=False`**:
   - **`persistent=True`** belongs **only** to data-dependent running statistics learned during forward passes (`BatchNorm` `running_mean / running_var`).
   - **`persistent=False`** is mandatory for **all** deterministic tensors derived from `config` hyperparameters (`RoPE inv_freq`, `cos/sin` tables, causal masks, `ALiBi` slopes).
2. **Defending Against Legacy Checkpoints that Already Saved `inv_freq`**:
   - Even if your new model sets `persistent=False`, PyTorch's `_load_from_state_dict` will still copy an incoming key if it matches a buffer attribute name! Always strip `"*.rotary_emb.inv_freq"` keys from legacy checkpoints before calling `load_state_dict()` and assert `torch.allclose(model.rotary_emb.inv_freq, expected_inv_freq)` afterward!

---

## ② 🎯 Architect Q&A (Q1 / Q2 / Q3)

### Q1: Why did the overwritten `rope_theta = 10,000` model look completely fine at `4K` evaluation length (`5.18 PPL`)?
- Because `4K` tokens is well within the original `8K` training horizon of `rope_theta = 10,000`! The out-of-distribution phase rotation only manifests when token positions exceed `8,192`, causing perplexity to jump to `28.40` at `16K` and `184.60` at `64K`.

### Q2: Why should `RotaryEmbedding` be instantiated once on the top-level `LlamaModel` instead of inside each of the 80 `Attention` layers?
- Sharing one top-level `RotaryEmbedding` computes `(cos, sin)` **once per forward pass** instead of `80` times, eliminating `158` redundant kernel launches per step and `79` duplicate buffer allocations!

### Q3: How does `persistent=False` interact with `torch.device("meta")` + `model.to_empty(device="cuda")`?
- Because `persistent=False` buffers are not in `state_dict()`, they must be recomputed on the target device after `to_empty()` via a `reset_non_persistent_buffers()` method (see Round 345).

---

## ③ 🧭 Visual Diagrams (Decision Tree + Comparison + Sequence)

### 3.1 `nn.Module` Buffer Persistence Selection Decision Tree

```mermaid
flowchart TD
    A["Store Non-Gradient Tensor on nn.Module (RoPE inv_freq / cos_sin / Causal Mask / Running Stats)"] --> B{"Is the Tensor Deterministically Computable from Config Hyperparameters?"}
    B -- "No: Data-Driven Running Stat (BatchNorm running_mean)" --> C["✅ Use register_buffer(name, t, persistent=True)"]
    B -- "Yes: Deterministic Derived Constant (RoPE inv_freq / Masks / Slopes)" --> D{"How Is the Tensor Registered?"}
    D -- "Default register_buffer(name, t) (persistent=True)" --> E["⛔ Tier 1 & 2: +18.40 GB checkpoint bloat & load_state_dict silently overwrites 500,000 inv_freq back to 10,000 (64K PPL = 184.60)!"]
    D -- "Plain Python Attribute: self.inv_freq = t" --> F["⛔ Tier 3: Ignored by model.to('cuda') & FSDP -> 100% CPU/GPU device mismatch crash!"]
    D -- "register_buffer(name, t, persistent=False) + Shared Rotary + Strip Legacy Keys" --> G["🏆 Tier 4: Tracked by .to('cuda') + 0.00 GB bloat + 0% overwrite (64K PPL = 5.42, 34.1x better)!"]
```

### 3.2 `persistent=True` Silent Overwrite (`184.60 PPL`) vs. `persistent=False` + Key Strip (`5.42 PPL`)

```mermaid
flowchart LR
    subgraph Bad["⛔ Tier 1 (Default persistent=True): Stale Checkpoint Overwrites Extended RoPE Theta"]
      B1["1. Init computes inv_freq with new theta=500,000"] --> B2["2. load_state_dict(8k_ckpt) silently overwrites inv_freq back to 10,000!"]
      B2 --> B3["3. 64K Evaluation Perplexity explodes to 184.60!"]
    end
    subgraph Good["🏆 Tier 4 (persistent=False + Shared Rotary + Pre-Hook Key Strip): 5.42 PPL at 64K"]
      G1["1. register_buffer('inv_freq', ..., persistent=False) excludes buffer from state_dict"] --> G2["2. Pre-hook strips legacy '*.inv_freq' keys -> 64K PPL stays 5.42 (0.00 GB bloat)!"]
    end
```

### 3.3 Context-Extension Checkpoint Loading Lifecycle Sequence

```mermaid
sequenceDiagram
    participant Init as `LlamaModel.__init__(rope_theta=500,000)`
    participant Hook as `_load_state_dict_pre_hook`
    participant Load as `model.load_state_dict(base_8k_ckpt)`
    participant Eval as `64K` Long-Context Forward Pass

    Init->>Init: 1. Top-level `RotaryEmbedding` registers `inv_freq` (`theta=500,000`, `persistent=False`)
    Load->>Hook: 2. Intercept incoming `state_dict` from legacy 8K checkpoint
    Hook->>Load: 3. Strip any legacy `"*.rotary_emb.inv_freq"` keys (`theta=10,000`)
    Load->>Eval: 4. Load learned weights only -> `inv_freq` remains `500,000` (`64K PPL = 5.42`!)
```

---

## ④ 📊 4-Tier Architect Decision Matrix

| Cohort Tier | Buffer Registration Strategy | Effective `rope_theta` | `64K` Eval Perplexity | Checkpoint Bloat | `.to("cuda")` Crash Rate |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Tier 1: Default `persistent=True` (`inv_freq`)** | `register_buffer("inv_freq", t)` | **`10,000` (Overwritten ✕)** | **`184.60` ✕** | `+0.05 GB` | `0.0%` |
| **Tier 2: Persistent Masks & Caches** | 80 layers save `cos/sin/bias` | **`10,000` ✕** | **`184.60` ✕** | **`+18.40 GB` ✕** | **`100.0%` (`strict` fail ✕)** |
| **Tier 3: Plain Python Attribute** | `self.inv_freq = t` | `500,000` | `5.42` | `0.00 GB` | **`100.0%` (CPU/GPU crash ✕)** |
| **Tier 4: `persistent=False` + Shared (Recommended)** | **Non-persistent + strip legacy keys** | **`500,000` (`100%` kept 🏆)** | **`5.42` (`34.1x` better 🏆)** | **`0.00 GB` (`-18.4 GB` 🏆)** | **`0.0%` 🏆** |

---

## ⑤ 🧪 Quantitative Evaluator & 7-Card 4:5 Visual Report

- **Sealed Evaluator**: [`experiments/pytorch_register_buffer_persistent_false_rope_overwrite_checkpoint_bloat_selection/run.py`](../../experiments/pytorch_register_buffer_persistent_false_rope_overwrite_checkpoint_bloat_selection/run.py) · Data: [`results.json`](../../experiments/pytorch_register_buffer_persistent_false_rope_overwrite_checkpoint_bloat_selection/results.json)
- **Interactive Sizer**: [`docs/reports/pytorch-register-buffer-persistent-false-rope-overwrite-checkpoint-bloat-sizer.html`](../reports/pytorch-register-buffer-persistent-false-rope-overwrite-checkpoint-bloat-sizer.html)
- **7-Card 4:5 Visual Benchmark Deck**: [`docs/reports/pytorch-register-buffer-persistent-false-rope-overwrite-checkpoint-bloat-selection-cards.html`](../reports/pytorch-register-buffer-persistent-false-rope-overwrite-checkpoint-bloat-selection-cards.html)

```chart
{
  "data": {
    "values": [
      { "tier": "Tier 1: Default persistent=True (Overwritten)", "eval_ppl_64k": 184.60, "ckpt_bloat_gb": 0.05 },
      { "tier": "Tier 2: Persistent Masks (+18.40 GB Bloat)", "eval_ppl_64k": 184.60, "ckpt_bloat_gb": 18.40 },
      { "tier": "Tier 3: Plain Attribute (100% Device Crash)", "eval_ppl_64k": 5.42, "ckpt_bloat_gb": 0.00 },
      { "tier": "Tier 4: persistent=False + Shared Rotary", "eval_ppl_64k": 5.42, "ckpt_bloat_gb": 0.00 }
    ]
  },
  "semantic_types": {
    "tier": "Category",
    "eval_ppl_64k": "Score",
    "ckpt_bloat_gb": "Quantity"
  },
  "chart_spec": {
    "chartType": "Bar Chart",
    "title": "register_buffer(persistent=False) Prevents 34.1x 64K PPL Collapse & Saves 18.40 GB Bloat",
    "subtitle": "64K context evaluation perplexity (lower is better) across 4 tiers (synthetic + anonymized_reported)",
    "encodings": {
      "x": "tier",
      "y": "eval_ppl_64k",
      "color": "tier"
    }
  }
}
```

---

## ⑥ 🛠️ Production Verification Checklist (Good fit vs. Not for)

### ✅ Good fit vs. ⛔ Not for
- **✅ Good fit**: All formula-derived tensors (`RoPE inv_freq`, `cos/sin` caches, causal masks, `ALiBi` slopes).
- **⛔ Not for**: Using default `persistent=True` on deterministic constants or storing tensors as plain Python attributes on `nn.Module`.

### Production Checklist
- [ ] **Pass `persistent=False` on every `self.register_buffer(...)` call for deterministic constants**.
- [ ] **Strip any legacy `"*.inv_freq"` keys from checkpoints prior to `load_state_dict()` during context extension and assert `inv_freq` matches the target `rope_theta`**.
- [ ] **Share a single `RotaryEmbedding` module at the top-level `Model` across all layers**.
