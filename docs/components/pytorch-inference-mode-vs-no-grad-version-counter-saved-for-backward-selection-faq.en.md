# PyTorch `torch.inference_mode()` vs. `torch.no_grad()` Version-Counter Dispatch Overhead (`1.66x` Slower Across `960` Ops) & Hybrid Training `SavedVariable` Backward Crash Selection Guide: Why `no_grad()` Still Allocates Atomic `VariableVersion` Counters, Why Raw `inference_mode()` Outputs Crash `nn.Linear` `100%` of the Time & How a Single Exit-Boundary `.clone()` (`2.92 ms`, `0%` Crash) Solves It

> **Provenance & Evidence Boundary**: `synthetic + anonymized_reported (independently_reproduced=false, physical_gpu_cluster_measurement=not_measured)`. All quantitative metrics in this guide are grounded in the sealed deterministic evaluator `experiments/pytorch_inference_mode_vs_no_grad_version_counter_saved_for_backward_selection/run.py` and `results.json`.

![PyTorch torch.inference_mode() vs torch.no_grad() & SavedVariable Crash Visual Summary](../../assets/svg/pytorch-inference-mode-vs-no-grad-version-counter-saved-for-backward-selection-faq-summary.svg)

---

## 0. Daily Capability Breakthrough

- **Today's Highlight Breakthrough**: **`c10::InferenceMode` Zero-Counter Internal Dispatch Paired with a Single Exit-Boundary `out = out.clone()` Bridge for Hybrid Frozen-to-Trainable Pipelines (`vLLM / SGLang / Qwen2.5-VL Frozen ViT / verl RL Rollout`)**.
- **Architect Value**: Across LLM serving engines, multimodal VLM fine-tuning (`Frozen ViT -> Trainable MM-Projector`), and co-located reinforcement learning (`Rollout -> Actor Trainer`), engineers routinely encounter two seemingly contradictory failure modes around `torch.no_grad()` and `torch.inference_mode()`:
  1. **The `1.66x` CPU Dispatcher Tax of `torch.no_grad()` (Tier 1 — `4.85 ms` vs `2.90 ms` Across `960` Dispatched Ops)**:
     - While `with torch.no_grad():` disables building the backward `grad_fn` graph, **every tensor created inside `no_grad()` still allocates a heap-backed atomic `c10::VariableVersion` (`version_counter_`) and executes `ADInplaceOrView` view-tracking kernels (`DifferentiableViewMeta`)** on every `.view()`, `.slice()`, `.transpose()`, and in-place operation—just in case that tensor is later used by autograd outside `no_grad()`! Across `960` op dispatches per step, this bookkeeping wastes **`1.95 ms/step`** of CPU dispatch overhead!
  2. **The `100%` `SavedVariable` Backward Crash When Passing Raw `@torch.inference_mode()` Outputs into Trainable Layers (`nn.Linear`, Tier 2)**:
     - Tensors created inside `torch.inference_mode()` have `is_inference() == True` and **no `VariableVersion` counter**.
     - When a frozen Vision Encoder (`ViT`) decorated with `@torch.inference_mode()` passes its output `vit_embeds` (`requires_grad=False`) into a trainable `mm_projector = nn.Linear(1280, 4096)` (`weight.requires_grad=True`), `nn.Linear` must save `vit_embeds` via `ctx.save_for_backward(input, weight)` to compute $\text{d}W = \text{d}Y^\top X_{\text{input}}$ during `.backward()`.
     - Because `vit_embeds` is an inference tensor without a version counter, PyTorch crashes **`100%` of the time** with:
       `RuntimeError: Inference tensors cannot be saved for backward. To work around you can make a clone to get a normal tensor and use it in autograd.`!
  3. **The `.detach()` Illusion (Tier 3 — Still Crashes `100%`!)**:
     - Calling `vit_embeds.detach()` or exiting the `with torch.inference_mode():` block does **not** clear `is_inference() == True` on an already-created tensor!
  By **running all `960` internal ops of the frozen backbone or rollout inside `with torch.inference_mode():` and executing a single `vit_embeds = vit_embeds.clone()` immediately *outside* the `inference_mode()` block (Tier 4)**, `960` version-counter allocations are reduced to **`1`** (`+0.02 ms` for a `16 MB` boundary tensor), achieving a **`1.66x` CPU dispatch speedup (`2.92 ms` vs `4.85 ms`)** with **`0.0%` crash rate**!

---

## ① 📌 Executive Summary

1. **First-Principles Dispatcher Mechanics (`c10::AutoGradMode` vs `c10::InferenceMode`)**:
   - `torch.no_grad()` removes only the `Autograd` dispatch key while keeping `ADInplaceOrView` active (`960` atomic version counters allocated per step).
   - `torch.inference_mode()` removes **both** `Autograd` and `ADInplaceOrView` from the thread-local dispatch key set (`0` version counters, `0` view metadata).
2. **Why `.clone()` Must Be Called *Outside* `inference_mode()` at the Boundary**:
   - Calling `out = out.clone()` after exiting `with torch.inference_mode():` allocates a fresh `TensorImpl` under normal autograd dispatch rules (`out.is_inference() == False`), making it `100%` safe for downstream `save_for_backward` and in-place mutations!

---

## ② 🎯 Architect Q&A (Q1 / Q2 / Q3)

### Q1: Why does `nn.Linear(x)` save `x` for backward even when `x.requires_grad == False`?
- Because for $Y = X W^\top$, the gradient with respect to the **trainable weight matrix $W$** is $\frac{\partial L}{\partial W} = \left(\frac{\partial L}{\partial Y}\right)^\top X$! Even though $\frac{\partial L}{\partial X}$ is not needed, computing $\frac{\partial L}{\partial W}$ requires the forward activation $X$, so `nn.Linear` calls `ctx.save_for_backward(x)` whenever `weight.requires_grad` is `True`!

### Q2: Why does mutating a rollout tensor in-place (`rollout_ids[mask] = pad_id`) fail after exiting `inference_mode()`?
- PyTorch forbids in-place modifications on `is_inference() == True` tensors outside an `InferenceMode` guard (`RuntimeError: Inplace update to inference tensor outside InferenceMode is not allowed`). Calling `.clone()` on the returned rollout tensor outside `inference_mode()` resolves this completely.

### Q3: Can we selectively re-enable normal tensor creation inside a function decorated with `@torch.inference_mode()`?
- Yes! Wrap the return statement in `with torch.inference_mode(False): return out.clone()`.

---

## ③ 🧭 Visual Diagrams (Decision Tree + Comparison + Sequence)

### 3.1 `torch.inference_mode()` vs. `torch.no_grad()` Selection Decision Tree

```mermaid
flowchart TD
    A["Execute Gradient-Free Forward Pass (960 Dispatched Ops: Serving / Frozen ViT / RL Rollout)"] --> B{"Does Output Enter a Trainable Layer (nn.Linear) or In-Place Mutation Outside?"}
    B -- "No: Pure Inference Server (vLLM / SGLang Worker)" --> C["🏆 Use @torch.inference_mode() End-to-End (2.90 ms, 1.67x faster, 0 version counters!)"]
    B -- "Yes: Hybrid Pipeline (Frozen ViT -> Trainable Projector, or RL Rollout -> Trainer)" --> D{"How Are Context & Exit Boundary Handled?"}
    D -- "Fallback to with torch.no_grad(): Everywhere" --> E["⚠️ Tier 1: Allocates 960 VariableVersion counters & ViewMeta -> 4.85 ms (1.66x slower)!"]
    D -- "Feed Raw @torch.inference_mode() (or .detach()) into nn.Linear" --> F["⛔ Tier 2 & 3: 100% RuntimeError crash when nn.Linear calls save_for_backward(input)!"]
    D -- "Run Frozen Ops Inside with torch.inference_mode(): + Call out.clone() Outside" --> G["🏆 Tier 4: 0 internal counters + 1 boundary .clone() (2.92 ms, 1.66x faster, 0.0% crash!)"]
```

### 3.2 `no_grad()` Overhead & Raw `inference_mode` Crash vs. Boundary `.clone()` Bridge

```mermaid
flowchart LR
    subgraph Bad["⛔ Tier 1 & Tier 2: 960 Version Counters (4.85 ms) or 100% SavedVariable Crash"]
      B1["Tier 1 (no_grad): Allocates 960 VariableVersion + ViewMeta (4.85 ms)"] --> B2["Tier 2 (Raw inference_mode): is_inference=True fed to Linear -> 100% Crash!"]
    end
    subgraph Good["🏆 Tier 4: Internal inference_mode() + Single Exit-Boundary .clone() (2.92 ms, 0% Crash)"]
      G1["with torch.inference_mode(): Run 32-layer ViT (960 ops, 0 counters, 2.90 ms)"] --> G2["Outside block: vit_embeds = vit_embeds.clone() (+0.02 ms) -> Safe for Linear!"]
    end
```

### 3.3 Frozen ViT to Trainable Projector Execution Sequence

```mermaid
sequenceDiagram
    participant ViT as Frozen ViT (`with torch.inference_mode():`)
    participant Bridge as Exit Boundary (`Outside inference_mode`)
    participant Proj as Trainable `mm_projector` (`nn.Linear`)
    participant Bwd as Autograd `.backward()` (`dW = dY^T @ X`)

    ViT->>ViT: 1. Execute 960 ops with `ADInplaceOrView` disabled (`0` version counters, `2.90 ms`)
    ViT->>Bridge: 2. Return `raw_embeds` (`is_inference() == True`)
    Bridge->>Proj: 3. `clean_embeds = raw_embeds.clone()` (`is_inference() == False`, `1` version counter, `+0.02 ms`)
    Proj->>Bwd: 4. `ctx.save_for_backward(clean_embeds)` succeeds (`0.0%` crash!) & computes `dW`
```

---

## ④ 📊 4-Tier Architect Decision Matrix

| Cohort Tier | Frozen / Rollout Execution Strategy | CPU Dispatch Latency | Version Counters / Step | `save_for_backward` Crash Rate | Autograd Safe |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Tier 1: `torch.no_grad()` Everywhere** | `with torch.no_grad():` across 960 ops | **`4.85 ms` (`1.00x` ✕)** | **`960` ✕** | `0.0%` | ✅ Yes (Slow dispatch) |
| **Tier 2: Raw `@torch.inference_mode()`** | Directly feed output to trainable `Linear` | `2.90 ms` | `0` | **`100.0%` (`RuntimeError` ✕)** | **⛔ 100% Crash** |
| **Tier 3: Exit `.detach()` Without `.clone()`** | Call `out.detach()` after `inference_mode` | `2.91 ms` | `0` | **`100.0%` (`is_inference=True` ✕)** | **⛔ 100% Crash** |
| **Tier 4: `inference_mode` + Exit `.clone()` (Recommended)** | **Internal `inference_mode` + outside `.clone()`** | **`2.92 ms` (`1.66x` faster 🏆)** | **`1` (`-99.9%` 🏆)** | **`0.0%` 🏆** | **✅ 100% Safe 🏆** |

---

## ⑤ 🧪 Quantitative Evaluator & 7-Card 4:5 Visual Report

- **Sealed Evaluator**: [`experiments/pytorch_inference_mode_vs_no_grad_version_counter_saved_for_backward_selection/run.py`](../../experiments/pytorch_inference_mode_vs_no_grad_version_counter_saved_for_backward_selection/run.py) · Data: [`results.json`](../../experiments/pytorch_inference_mode_vs_no_grad_version_counter_saved_for_backward_selection/results.json)
- **Interactive Sizer**: [`docs/reports/pytorch-inference-mode-vs-no-grad-version-counter-saved-for-backward-sizer.html`](../reports/pytorch-inference-mode-vs-no-grad-version-counter-saved-for-backward-sizer.html)
- **7-Card 4:5 Visual Benchmark Deck**: [`docs/reports/pytorch-inference-mode-vs-no-grad-version-counter-saved-for-backward-selection-cards.html`](../reports/pytorch-inference-mode-vs-no-grad-version-counter-saved-for-backward-selection-cards.html)

```chart
{
  "data": {
    "values": [
      { "tier": "Tier 1: torch.no_grad() (960 Version Counters)", "cpu_dispatch_ms": 4.85, "crash_pct": 0.0 },
      { "tier": "Tier 2: Raw inference_mode (100% Crash)", "cpu_dispatch_ms": 2.90, "crash_pct": 100.0 },
      { "tier": "Tier 3: .detach() Without .clone() (100% Crash)", "cpu_dispatch_ms": 2.91, "crash_pct": 100.0 },
      { "tier": "Tier 4: inference_mode + Exit Boundary .clone()", "cpu_dispatch_ms": 2.92, "crash_pct": 0.0 }
    ]
  },
  "semantic_types": {
    "tier": "Category",
    "cpu_dispatch_ms": "Quantity",
    "crash_pct": "Percentage"
  },
  "chart_spec": {
    "chartType": "Bar Chart",
    "title": "inference_mode() + Exit Boundary .clone() Speeds Up CPU Dispatch 1.66x with 0% Crash Rate",
    "subtitle": "CPU C++ dispatcher latency (ms) across 960 ops/step (synthetic + anonymized_reported)",
    "encodings": {
      "x": "tier",
      "y": "cpu_dispatch_ms",
      "color": "tier"
    }
  }
}
```

---

## ⑥ 🛠️ Production Verification Checklist (Good fit vs. Not for)

### ✅ Good fit vs. ⛔ Not for
- **✅ Good fit**: Pure LLM serving workers (`@torch.inference_mode()`), frozen VLM vision/audio towers with a single exit `.clone()` before the trainable projector, and RL rollout-to-trainer handoffs.
- **⛔ Not for**: Using `torch.no_grad()` in pure inference servers, or feeding uncloned `is_inference() == True` tensors into trainable modules or in-place mutations.

### Production Checklist
- [ ] **Replace `@torch.no_grad()` with `@torch.inference_mode()` in all pure inference and evaluation workers**.
- [ ] **Call `out = out.clone()` immediately *outside* `with torch.inference_mode():` whenever the output enters a trainable module (`nn.Linear`) or undergoes in-place mutation**.
- [ ] **Assert `assert not tensor.is_inference()` at the entrypoint of trainable adapter/projector modules**.
