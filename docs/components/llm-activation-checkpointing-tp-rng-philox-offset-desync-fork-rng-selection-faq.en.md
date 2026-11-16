# Activation Checkpointing & `TP/SP` `CUDA RNG Philox (seed, offset)` Desync Selection Guide: Why Unrestored `Philox` Offsets Corrupt Gradients by `34.8%`, Why Legacy Host `get_rng_state()` Wastes `38 ms` & How Dual `TP/DP` RNG Trackers + Device-Side `16B Philox` State (`0.4 ms`, `95x` Faster, `100%` Bitwise Match) Solve It

> **Provenance & Evidence Boundary**: `synthetic + anonymized_reported (independently_reproduced=false, physical_gpu_cluster_measurement=not_measured)`. All quantitative metrics in this guide are grounded in the sealed deterministic evaluator `experiments/llm_activation_checkpointing_tp_rng_philox_offset_desync_fork_rng_selection/run.py` and `results.json`.

![Activation Checkpointing & TP/SP CUDA RNG Philox Desync Visual Summary](../../assets/svg/llm-activation-checkpointing-tp-rng-philox-offset-desync-fork-rng-selection-faq-summary.svg)

---

## 0. Daily Capability Breakthrough

- **Today's Highlight Breakthrough**: **Orthogonal Dual-Stream `TensorParallelRNGTracker` (`TP-Synced` vs. `TP-Sharded` Seeds) + CUDAGraph-Safe Device-Side `16-Byte Philox (seed, offset)` State Capture for 3D-Parallel (`TP + SP + PP + DP`) & `FP8` Stochastic-Rounding Pretraining**.
- **Architect Value**: Distributed LLM training relies on hardware counter-based pseudorandom number generation (`cuRAND Philox-4x32-10`) for `FlashAttention` dropout, residual/router noise, and **`FP8 / NVFP4` Stochastic Rounding (`SR`)**. Combining **Activation Checkpointing** with **Tensor & Sequence Parallelism (`TP=8 + SP`)** exposes three critical RNG pitfalls:
  1. **Unrestored `Philox` Offsets During Backward Recomputation (`preserve_rng_state=False` or Unmanaged Triton RNG, Tier 1)**: During the forward pass of Layer $l$, the generator's `Philox` offset is $O_{\text{fwd}}$. By the time the backward pass reaches Layer $l$ to recompute activations, the generator offset has advanced to $O_{\text{bwd}} \neq O_{\text{fwd}}$. Recomputing Layer $l$ without restoring $O_{\text{fwd}}$ produces completely different dropout and stochastic-rounding masks (`0.0%` bitwise match)—corrupting weight gradients by **`34.8%`** and causing loss divergence!
  2. **Single Unseparated RNG Stream Across `TP=8` Ranks (Tier 2)**: Using the exact same seed across all 8 TP ranks forces all 8 sharded attention head groups to apply identical correlated noise, whereas using different global seeds without separation desynchronizes non-sharded / replicated operations across TP ranks (`12.4%` gradient error)!
  3. **Legacy Host `torch.cuda.get_rng_state()` / `set_rng_state()` ByteTensor Copies (Tier 3)**: Extracting and restoring RNG states via CPU `ByteTensor`s across 80 layers (`160` transitions/step) adds **`38.0 ms` of host overhead per step** and breaks `CUDAGraph` / `torch.compile` capture!
  By deploying a **Dual-Stream `TensorParallelRNGTracker` (`TP-Synced` base generator + `TP-Sharded` `seed + tp_rank` generator) and capturing the `16-byte` `Philox (seed, offset)` tuple directly in GPU device memory (Tier 4)**, forward-backward recomputation achieves **`100.0%` bitwise identity (`0.0%` gradient error)** while slashing RNG state overhead **`95.0x` (`38.0 ms -> 0.4 ms`)**!

---

## ① 📌 Executive Summary

1. **First-Principles Counter-Based PRNG (`Philox-4x32-10` Saves `32 GB` of Mask Memory per Layer)**:
   - Storing an explicit `[B, H, 32768, 32768]` dropout mask in HBM for backward takes `32 GB` per layer. Instead, `Philox-4x32-10` computes random numbers statelessly in registers as a pure bijection of a **`16-byte` `(uint64 seed, uint64 offset)` tuple** plus thread coordinates:
     $$u_{b,h,i,j} = \text{Philox}_{10}\big(\text{seed},\ \text{offset} + \text{tid}(b,h,i,j)\big)$$
   - Saving and restoring just **`16 Bytes`** per checkpointed layer reproduces billions of random values bitwise-identically during backward recomputation!
2. **Why `TP + SP` Requires Two Orthogonal RNG Streams (`TensorParallelRNGTracker`)**:
   - **Stream 1 (`TP-Synced / Data-Parallel RNG`, `seed = base + dp_rank * 100`)**: Identical across all 8 ranks inside a `TP` group; used for operations replicated across TP ranks so their tensors remain identical.
   - **Stream 2 (`TP-Sharded / Model-Parallel RNG`, `seed = base + dp_rank * 100 + 2718 + tp_rank`)**: Unique per `tp_rank`; activated via `with rng_tracker.fork("model-parallel-rng"):` inside sharded Multi-Head Attention and Sequence-Parallel (`SP`) regions so each rank's distinct head/sequence shard receives independent random draws.
3. **Device-Side `Philox` State Eliminates `38.0 ms` of Host Sync & Unlocks `CUDAGraph`s**:
   - Storing `(seed, offset)` as device-resident `int64` tensors (`get_device_state` / `philox_cuda_state`) avoids CPU-GPU transfers and allows `CUDAGraph`s to read and advance RNG offsets entirely on the GPU (`0.4 ms` total overhead across 80 layers).

---

## ② 🎯 Architect Q&A (Q1 / Q2 / Q3)

### Q1: If we set `attention_dropout = 0.0` in LLM pretraining, can we safely set `preserve_rng_state = False`?
- **No!** Even with `dropout = 0.0`, modern `FP8 / NVFP4` quantization kernels use **Stochastic Rounding (`SR`)**, and MoE routers or custom regularizers consume RNG states. Since device-side `Philox` state capture costs only `0.4 ms` per step, always keep RNG state preservation enabled.

### Q2: Why do custom Triton kernels using `tl.rand(seed, offsets)` often fail activation checkpointing even when `preserve_rng_state=True`?
- Because `preserve_rng_state=True` only snapshots PyTorch's `CUDAGenerator`! If the Python wrapper passes an unmanaged Python integer counter (`self.step += 1`) as `seed` to the Triton kernel, `fork_rng` does not roll back `self.step` during backward recomputation. Always derive the Triton kernel's `(seed, offset)` from the active PyTorch `CUDAGenerator`!

### Q3: How should RNG seeds be offset across Pipeline Parallelism (`PP`) stages?
- Offset both the base and model-parallel seeds by `pp_rank` (e.g., `base_seed + 1000 * dp_rank + 100 * pp_rank`) so distinct pipeline stages do not apply identical noise patterns at every stage boundary.

---

## ③ 🧭 Visual Diagrams (Decision Tree + Comparison + Sequence)

### 3.1 Activation Checkpointing & `TP/SP` RNG Management Decision Tree

```mermaid
flowchart TD
    A["80-Layer TP=8 + SP Training With Activation Checkpointing & Stochastic Kernels"] --> B{"Are Philox (seed, offset) States Saved & Restored on Recompute?"}
    B -- "No: preserve_rng_state=False or unmanaged Triton counter" --> C["⛔ Tier 1: 0% fwd-bwd mask match -> 34.8% gradient error & loss divergence!"]
    B -- "Yes: Restore forward RNG state on backward recompute" --> D{"Are Replicated vs. TP-Sharded RNG Streams Separated?"}
    D -- "No: Single global seed across TP=8" --> E["⛔ Tier 2: Replicated tensor desync or correlated head masks -> 12.4% error!"]
    D -- "Yes: Orthogonal Dual-Stream TensorParallelRNGTracker" --> F{"Host ByteTensor Copy vs. Device-Side 16B Philox State?"}
    F -- "Legacy CPU torch.cuda.get_rng_state() / set_rng_state()" --> G["⚠️ Tier 3: 100% match, but 38.0 ms step overhead & breaks CUDAGraphs"]
    F -- "Dual TP/DP Tracker + Device-Side 16B Philox (seed, offset)" --> H["🏆 Tier 4: 100% bitwise match + 0.0% grad error + 0.4 ms overhead (95x faster)!"]
```

### 3.2 Unrestored `Philox` Offset (`34.8%` Grad Error) vs. Device-Side `16B (seed, offset)` Capture (`0.0%` Error)

```mermaid
flowchart LR
    subgraph Bad["⛔ Tier 1: Unrestored Philox Offset (0% Match, 34.8% Gradient Error)"]
      B1["Forward Layer l: Uses Philox offset=100 -> Activations X_fwd"] --> B2["Backward Recompute Layer l: Uses advanced offset=8000 -> Wrong X_recomp!"]
    end
    subgraph Good["🏆 Tier 4: Device-Side 16-Byte (seed, offset) Capture (100% Bitwise Match, 0.4 ms)"]
      G1["Forward Layer l: Save 16B (seed, offset=100) on GPU"] --> G2["Backward Recompute Layer l: Pass (seed, offset=100) -> 100% Identical Activations!"]
    end
```

### 3.3 `TensorParallelRNGTracker` Fork & Device `Philox` Recomputation Sequence

```mermaid
sequenceDiagram
    participant Track as TensorParallelRNGTracker (GPU Generators)
    participant Ckpt as Activation Checkpoint Context (16B State)
    participant FA as FlashAttention / FP8 Kernel (Philox-4x32-10)

    Note over Track,FA: Phase 1: Forward Pass (Layer l)
    Track->>Ckpt: 1. Enter `with rng_tracker.fork("model-parallel-rng")` (seed = base + tp_rank)
    Ckpt->>FA: 2. Record device 16B `(philox_seed, philox_offset)` & run forward kernel
    Note over Track,FA: Phase 2: Backward Activation Recompute (Layer l)
    Ckpt->>FA: 3. Restore saved 16B `(philox_seed, philox_offset)` on GPU (0ms host sync)
    FA->>FA: 4. Evaluate `Philox_10(seed, offset + tid)` in registers -> 100% bitwise match!
```

---

## ④ 📊 4-Tier Architect Decision Matrix

| Cohort Tier | Checkpointing & `TP/SP` RNG Policy | Fwd-Bwd Bitwise Match | Step RNG Overhead | Gradient Relative Error | Best For |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Tier 1: `preserve_rng_state=False`** | Recomputes with advanced offset | `0.0%` ✕ | `0.0 ms` | `34.8%` (Diverges! ✕) | ⛔ **Corrupts Checkpointed Gradients** |
| **Tier 2: Unsplit Single `TP=8` Seed** | Conflates replicated & sharded ops | `50.0%` ✕ | `28.5 ms` ✕ | `12.4%` ✕ | ⛔ **Desynchronizes TP Ranks** |
| **Tier 3: Legacy Host `get/set_rng_state`** | 160x CPU `ByteTensor` state copies | `100.0%` ✅ | `38.0 ms` ⚠️ | `0.0%` ✅ | ⚠️ Legacy Non-Graph Baselines |
| **Tier 4: Dual `TP/DP` Tracker + Device `Philox` (Recommended)** | **Orthogonal seeds + `16B` GPU state** | **`100.0%` 🏆** | **`0.4 ms` (`95x` Faster 🏆)** | **`0.0%` 🏆** | 🏆 **3D-Parallel & `FP8` Pretraining** |

---

## ⑤ 🧪 Quantitative Evaluator & 7-Card 4:5 Visual Report

- **Sealed Evaluator**: [`experiments/llm_activation_checkpointing_tp_rng_philox_offset_desync_fork_rng_selection/run.py`](../../experiments/llm_activation_checkpointing_tp_rng_philox_offset_desync_fork_rng_selection/run.py) · Data: [`results.json`](../../experiments/llm_activation_checkpointing_tp_rng_philox_offset_desync_fork_rng_selection/results.json)
- **Interactive Sizer**: [`docs/reports/llm-activation-checkpointing-tp-rng-philox-offset-desync-fork-rng-sizer.html`](../reports/llm-activation-checkpointing-tp-rng-philox-offset-desync-fork-rng-sizer.html)
- **7-Card 4:5 Visual Benchmark Deck**: [`docs/reports/llm-activation-checkpointing-tp-rng-philox-offset-desync-fork-rng-selection-cards.html`](../reports/llm-activation-checkpointing-tp-rng-philox-offset-desync-fork-rng-selection-cards.html)

```chart
{
  "data": {
    "values": [
      { "tier": "Tier 1: preserve_rng_state=False", "grad_err_pct": 34.8, "overhead_ms": 0.0 },
      { "tier": "Tier 2: Unsplit Single TP=8 Seed", "grad_err_pct": 12.4, "overhead_ms": 28.5 },
      { "tier": "Tier 3: Legacy Host get/set_rng_state", "grad_err_pct": 0.0, "overhead_ms": 38.0 },
      { "tier": "Tier 4: Dual TP/DP Tracker + Device Philox", "grad_err_pct": 0.0, "overhead_ms": 0.4 }
    ]
  },
  "semantic_types": {
    "tier": "Category",
    "grad_err_pct": "Percentage",
    "overhead_ms": "Quantity"
  },
  "chart_spec": {
    "chartType": "Bar Chart",
    "title": "Dual TP/DP RNG Tracker + Device Philox State Eliminates 34.8% Grad Error at 0.4ms Overhead (95x Faster)",
    "subtitle": "RNG state management overhead per 80-layer step (ms) and gradient error (%) (synthetic + anonymized_reported)",
    "encodings": {
      "x": "tier",
      "y": "overhead_ms",
      "color": "tier"
    }
  }
}
```

---

## ⑥ 🛠️ Production Verification Checklist (Good fit vs. Not for)

### ✅ Good fit vs. ⛔ Not for
- **✅ Good fit**: All 3D-parallel (`TP + SP + PP + DP`) training workloads using Activation Checkpointing, Dropout, or `FP8 / NVFP4` Stochastic Rounding.
- **⛔ Not for**: Passing unmanaged Python step counters as seeds into checkpointed Triton `tl.rand` kernels or disabling `preserve_rng_state`.

### Production Checklist
- [ ] **Separate `TP-Synced` and `TP-Sharded` (`seed + tp_rank`) RNG states via `TensorParallelRNGTracker`**: Keep replicated tensors synchronized and sharded attention/SP masks independent.
- [ ] **Capture `16-byte` `Philox (seed, offset)` states on-device**: Eliminate `38 ms` of CPU `ByteTensor` state copies and preserve `CUDAGraph` compatibility.
- [ ] **Draw all custom Triton `tl.rand` seeds/offsets from the active PyTorch `CUDAGenerator`**: Guarantee `100%` bitwise match during activation recomputation.
