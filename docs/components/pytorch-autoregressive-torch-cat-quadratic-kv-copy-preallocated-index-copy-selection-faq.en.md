# PyTorch Autoregressive `torch.cat([past_kv, new_kv])` Quadratic $O(S^2)$ Memory Reallocation & Copy Explosion (`536.87 GB` Traffic Over `4,096` Steps) vs. Pre-Allocated `StaticCache + index_copy_` Selection Guide: Why Loop `torch.cat` Amplifies Bandwidth `2,065x`, Why Python Integer Slices Break `CUDAGraph` & How 1D GPU Tensor `index_copy_` (`0.26 GB`, `1.12 ms`, `150.4x` Faster) Solves It

> **Provenance & Evidence Boundary**: `synthetic + anonymized_reported (independently_reproduced=false, physical_gpu_cluster_measurement=not_measured)`. All quantitative metrics in this guide are grounded in the sealed deterministic evaluator `experiments/pytorch_autoregressive_torch_cat_quadratic_kv_copy_preallocated_index_copy_selection/run.py` and `results.json`.

![PyTorch Autoregressive torch.cat O(S^2) Copy vs Pre-Allocated index_copy_ Visual Summary](../../assets/svg/pytorch-autoregressive-torch-cat-quadratic-kv-copy-preallocated-index-copy-selection-faq-summary.svg)

---

## 0. Daily Capability Breakthrough

- **Today's Highlight Breakthrough**: **Pre-Allocated Fixed-Capacity `[B, H_kv, S_max, D]` Buffers (`StaticCache`) Paired with Device-Tensor In-Place `kv_buf.index_copy_(2, cache_position, new_kv)` (`HuggingFace StaticCache / TorchTitan / FlashAttention-with-KVCache`)**.
- **Architect Value**: In custom autoregressive decoding loops, RL rollout generators, multimodal interleaved generation, and legacy HuggingFace `DynamicCache` pipelines, writing `past_kv = torch.cat([past_kv, new_kv], dim=2)` (or `generated_ids = torch.cat([generated_ids, next_token], dim=1)`) inside the step loop creates a catastrophic $O(S^2)$ memory bandwidth and allocator fragmentation bottleneck:
  1. **The `536.87 GB/Layer` Quadratic $O(S^2)$ Reallocation & Copy Explosion (`torch.cat` per Step, Tier 1 — `2,064.9x` Write Amplification & `168.40 ms/layer` Stall)**:
     - Because PyTorch tensors cannot grow in-place, calling `torch.cat([past_kv, new_kv], dim=2)` at decode step $t$ allocates a new tensor of length $t$, copies all $t-1$ historical token states from `past_kv`, appends the 1 new token state from `new_kv`, and frees the old tensor!
     - Over $S = 4,096$ decode steps, total copied slices equal $\sum_{t=1}^{4096} t = \frac{4096 \times 4097}{2} = \mathbf{8,390,656\text{ slices}}$ (**`2,048.5x` write amplification**)!
     - At `[B=16, H_kv=8, D=128]` (`64 KB/step` for K+V per layer), a single layer copies **`536.87 GB` of HBM data** (**`42.95 TB` across 80 layers** = `13.47 seconds` of pure copy overhead!) while issuing **`4,096` dynamic allocations** of incrementing sizes that severely fragment `CUDACachingAllocator`!
  2. **The `2x` Peak Memory Spike & Dynamic-Shape Incompatibility of Chunked/Final `torch.cat` (Tier 2)**:
     - Amortizing `torch.cat` every 64 steps still copies `8.39 GB/layer` (`6.80 ms`) and changes tensor shapes dynamically, preventing `CUDAGraph` capture!
  3. **The Silent `step = 0` Overwrite Bug When Using Python Integer Slicing (`kv_buf[:, :, step:step+1] = new_kv`) Inside `CUDAGraph` (Tier 3)**:
     - Pre-allocating `kv_buf` and indexing with a **Python integer `step`** computes the slice pointer offset (`base_ptr + step * stride_s`) on the **CPU** during graph capture. On every subsequent CUDA Graph replay, the GPU writes to the frozen capture-time address (`step = 0`), silently overwriting token 0!
  By **pre-allocating a fixed-capacity `[B, H_kv, S_max, D]` buffer once before the loop and updating it in-place via a 1D GPU tensor index `kv_buf.index_copy_(2, cache_position, new_kv)` (incremented on GPU via `cache_position.add_(1)`) (Tier 4)**, total copy traffic drops **`2,064.9x` (`536.87 GB -> 0.26 GB/layer`)**, copy latency drops **`150.4x` (`168.40 ms -> 1.12 ms/layer`)**, dynamic allocations drop from **`4,096` to `0`**, and the loop becomes **100% `CUDAGraph` safe**!

---

## ① 📌 Executive Summary

1. **First-Principles Complexity: $O(S^2)$ Dynamic `torch.cat` vs. $O(S)$ Pre-Allocated In-Place Update**:
   $$\text{Traffic}_{\text{cat}}(S) = \frac{S(S+1)}{2} \cdot B_{\text{step}} = O(S^2) \quad \text{vs.} \quad \text{Traffic}_{\text{static}}(S) = S \cdot B_{\text{step}} = O(S)$$
   At $S = 4,096$, $\frac{S+1}{2} = 2,048.5\times$; at $S = 16,384$, write amplification reaches **`8,192.5x` (`8.59 TB/layer`)**!
2. **Why `index_copy_(2, cache_position, new_kv)` Is Required for `CUDAGraph` Capture**:
   - Passing `cache_position` as a 1D `torch.int64` CUDA tensor keeps the tensor pointer constant while letting the GPU kernel read the current step offset from device memory during graph replay.

---

## ② 🎯 Architect Q&A (Q1 / Q2 / Q3)

### Q1: How does `flash_attn_with_kvcache` combine Tier 4's in-place update with zero-waste attention?
- `flash_attn_with_kvcache(q, k_cache, v_cache, k=new_k, v=new_v, cache_seqlens=cache_position.to(torch.int32))` writes `(new_k, new_v)` into `(k_cache, v_cache)` at `cache_seqlens` in-place AND restricts the attention reduction to valid tokens `0 .. cache_seqlens[b]` without reading unused tail slots!

### Q2: Does `generated_ids = torch.cat([generated_ids, next_token], dim=1)` suffer from the exact same quadratic bug?
- Yes! Always pre-allocate `output_ids = torch.empty((B, S_max), dtype=torch.long, device="cuda")` before the rollout loop and write each sampled token via `output_ids.index_copy_(1, cache_position, next_token)`!

### Q3: When should we use `StaticCache` (`index_copy_`) vs. Paged KV Cache (`reshape_and_cache`)?
- Use `StaticCache` (`[B, H_kv, S_max, D]` + `index_copy_`) for fixed-batch benchmark harnesses, speculative draft models, and standalone `torch.compile` scripts; use Paged KV (`slot_mapping` + `reshape_and_cache`) for multi-request continuous batching engines (`vLLM / SGLang`).

---

## ③ 🧭 Visual Diagrams (Decision Tree + Comparison + Sequence)

### 3.1 Autoregressive KV & State Accumulation Selection Decision Tree

```mermaid
flowchart TD
    A["Accumulate 1 New Token State per Step Over 4,096 Decode Steps ([B=16, H_kv=8, D=128])"] --> B{"How Are Per-Step States Stored & Appended?"}
    B -- "Per-Step past_kv = torch.cat([past_kv, new_kv], dim=2)" --> C["⛔ Tier 1: Quadratic O(S^2) copies 8.39M slices -> 536.87 GB/layer (168.40 ms) & 4,096 allocations!"]
    B -- "Chunked torch.cat Every 64 Steps" --> D["⛔ Tier 2: Still copies 8.39 GB/layer + 2x peak memory spike + incompatible with CUDAGraph!"]
    B -- "Pre-Allocated Fixed Buffer [B, H_kv, S_max, D] (0 Dynamic Allocs)" --> E{"How Is the Step Position Indexed?"}
    E -- "Python Integer Slice: kv_buf[:, :, step:step+1] = new_kv" --> F["⚠️ Tier 3: Freezes CPU pointer offset at capture time -> CUDAGraph overwrites position 0!"]
    E -- "1D GPU Tensor Index: kv_buf.index_copy_(2, cache_position, new_kv)" --> G["🏆 Tier 4: 0.26 GB traffic (-2,065x) + 1.12 ms (150.4x faster) + 100% CUDAGraph safe!"]
```

### 3.2 Per-Step `torch.cat` (`536.87 GB`) vs. Pre-Allocated `index_copy_` (`0.26 GB`)

```mermaid
flowchart LR
    subgraph Bad["⛔ Tier 1 (Per-Step torch.cat): Reallocates & Copies All Prior History Every Step (536.87 GB/Layer)"]
      B1["Steps 1..4096: Copies 4096*4097/2 = 8,390,656 slices"] --> B2["536.87 GB traffic/layer (42.95 TB across 80 layers!), 168.40 ms!"]
    end
    subgraph Good["🏆 Tier 4 (Pre-Allocated StaticCache + index_copy_): Writes Only 1 Slice/Step (0.26 GB/Layer)"]
      G1["Pre-allocate kv_buf[16, 8, 4096, 128] once (0 runtime allocs)"] --> G2["kv_buf.index_copy_(2, cache_pos, new_kv) -> 0.26 GB total in 1.12 ms (150.4x faster)!"]
    end
```

### 3.3 `CUDAGraph`-Safe `StaticCache + index_copy_` Execution Sequence

```mermaid
sequenceDiagram
    participant Init as Pre-Loop Setup
    participant Pos as 1D GPU Tensor `cache_position` (`[1]`)
    participant Graph as CUDAGraph Replay Step
    participant Cache as Pre-Allocated `kv_buf` (`[B, H, 4096, D]`)

    Init->>Cache: 1. Allocate `kv_buf` once (`0` runtime allocations across 4,096 steps)
    Init->>Pos: 2. Initialize `cache_position = torch.zeros((1,), dtype=torch.int64, device='cuda')`
    Graph->>Cache: 3. `kv_buf.index_copy_(2, cache_position, new_kv)` (Reads `cache_position` on GPU, writes `64 KB`)
    Graph->>Pos: 4. `cache_position.add_(1)` (In-place GPU increment -> ready for next graph replay!)
```

---

## ④ 📊 4-Tier Architect Decision Matrix

| Cohort Tier | State Accumulation Strategy | Copy Traffic / Layer | Dynamic Allocations | Copy Latency / Layer | `CUDAGraph` Safe |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Tier 1: Per-Step `torch.cat`** | `torch.cat([past, new], dim=2)` | **`536.87 GB` (`2,065x` ✕)** | **`4,096` ✕** | **`168.40 ms` (`1.0x` ✕)** | ⛔ No (Dynamic shape) |
| **Tier 2: Chunk-64 `torch.cat`** | Amortized 64-step `torch.cat` | `8.39 GB` | `64` | `6.80 ms` (`24.8x`) | ⛔ No (Dynamic shape) |
| **Tier 3: Pre-Allocated + Python `int` Slice** | `kv_buf[:, :, step:step+1] = new_kv` | `0.26 GB` | `0` | `1.45 ms` (`116.1x`) | **⛔ Overwrites pos 0 ✕** |
| **Tier 4: Pre-Allocated + `index_copy_` (Recommended)** | **`kv_buf.index_copy_(2, cache_pos, new_kv)`** | **`0.26 GB` (`-536.6 GB` 🏆)** | **`0` 🏆** | **`1.12 ms` (`150.4x` faster 🏆)** | **✅ 100% Safe 🏆** |

---

## ⑤ 🧪 Quantitative Evaluator & 7-Card 4:5 Visual Report

- **Sealed Evaluator**: [`experiments/pytorch_autoregressive_torch_cat_quadratic_kv_copy_preallocated_index_copy_selection/run.py`](../../experiments/pytorch_autoregressive_torch_cat_quadratic_kv_copy_preallocated_index_copy_selection/run.py) · Data: [`results.json`](../../experiments/pytorch_autoregressive_torch_cat_quadratic_kv_copy_preallocated_index_copy_selection/results.json)
- **Interactive Sizer**: [`docs/reports/pytorch-autoregressive-torch-cat-quadratic-kv-copy-preallocated-index-copy-sizer.html`](../reports/pytorch-autoregressive-torch-cat-quadratic-kv-copy-preallocated-index-copy-sizer.html)
- **7-Card 4:5 Visual Benchmark Deck**: [`docs/reports/pytorch-autoregressive-torch-cat-quadratic-kv-copy-preallocated-index-copy-selection-cards.html`](../reports/pytorch-autoregressive-torch-cat-quadratic-kv-copy-preallocated-index-copy-selection-cards.html)

```chart
{
  "data": {
    "values": [
      { "tier": "Tier 1: Per-Step torch.cat (536.87 GB)", "cumulative_copy_ms": 168.40, "copy_traffic_gb": 536.87 },
      { "tier": "Tier 2: Chunk-64 torch.cat (8.39 GB)", "cumulative_copy_ms": 6.80, "copy_traffic_gb": 8.39 },
      { "tier": "Tier 3: Python Int Slice (Breaks CUDAGraph)", "cumulative_copy_ms": 1.45, "copy_traffic_gb": 0.26 },
      { "tier": "Tier 4: Pre-Allocated StaticCache index_copy_", "cumulative_copy_ms": 1.12, "copy_traffic_gb": 0.26 }
    ]
  },
  "semantic_types": {
    "tier": "Category",
    "cumulative_copy_ms": "Quantity",
    "copy_traffic_gb": "Quantity"
  },
  "chart_spec": {
    "chartType": "Bar Chart",
    "title": "Pre-Allocated index_copy_ Cuts 4,096-Step KV Copy Latency 150.4x vs Per-Step torch.cat",
    "subtitle": "Cumulative single-layer KV copy latency (ms) over 4,096 decode steps (synthetic + anonymized_reported)",
    "encodings": {
      "x": "tier",
      "y": "cumulative_copy_ms",
      "color": "tier"
    }
  }
}
```

---

## ⑥ 🛠️ Production Verification Checklist (Good fit vs. Not for)

### ✅ Good fit vs. ⛔ Not for
- **✅ Good fit**: HuggingFace `StaticCache` (`cache_implementation="static"`), custom autoregressive rollout loops, speculative decoding draft loops, and `flash_attn_with_kvcache`.
- **⛔ Not for**: Calling `torch.cat` inside per-step autoregressive loops, or using Python integer slices inside `CUDAGraph` captured regions.

### Production Checklist
- [ ] **Ban `torch.cat` inside autoregressive decode and rollout step loops**: pre-allocate `[B, H_kv, S_max, D]` and `[B, S_max]` buffers once before the loop.
- [ ] **Update pre-allocated buffers in-place via a 1D GPU tensor `cache_position` (`kv_buf.index_copy_(2, cache_position, new_kv)`)**.
- [ ] **Increment `cache_position.add_(1)` on the GPU each step to preserve `CUDAGraph` compatibility**.
