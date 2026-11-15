# PyTorch `torch.compile` (`TorchInductor`), Dynamic Shape Guard Recompilation Storms, `CUDAGraph Trees` Memory Pool Bloat & Piecewise Compilation Selection Guide

> **Provenance & Evidence Tier**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`). Grounded in [`experiments/pytorch_compile_inductor_cudagraph_trees_dynamic_shape_selection/results.json`](../../experiments/pytorch_compile_inductor_cudagraph_trees_dynamic_shape_selection/results.json).
> **中文版本**: [PyTorch `torch.compile`（`TorchInductor`）、动态 Shape 重编译风暴（`Guard Explosion`）、`CUDAGraph Trees` 显存池膨胀与 `Piecewise` 分段编译选型指南](./pytorch-compile-inductor-cudagraph-trees-dynamic-shape-recompilation-guard-selection-faq.md)

![PyTorch torch.compile & Piecewise CUDAGraph Summary](../../assets/svg/pytorch-compile-inductor-cudagraph-trees-dynamic-shape-recompilation-guard-selection-faq-summary.svg)

---

## 0. 🔥 Daily Hottest Capability Breakthrough

- **`Piecewise Compilation` + `mark_dynamic` Symbolic Shapes + Shared-Pool Bucketed `CUDAGraph` Eliminating `8x` Online Recompilation Storms and `+18.4 GB` `CUDAGraph Trees` HBM Bloat (`6.8 ms` TPOT, `1.62x` Faster)**: Applying a naive full-model `torch.compile(model)` to variable-length LLM serving or SFT training causes two major production failures:
  1. **Dynamic Shape Guard Recompilation Storms & Silent Eager Fallback (`Tier 1`)**: As token counts vary per batch, un-marked dimensions trigger repeated Dynamo Guard failures (stalling online requests for `35–60 s` per recompile) until hitting `cache_size_limit = 8` and silently falling back to uncompiled Eager execution (`11.0 ms` TPOT)!
  2. **`mode="reduce-overhead"` (`CUDAGraph Trees`) Memory Pool Bloat (`Tier 2`)**: Wrapping dynamic branching or multi-bucket shapes inside automatic `CUDAGraph Trees` allocates disjoint private memory pools across graph paths, consuming **`18.4 GB` HBM** and starving KV Cache capacity.
  3. **Piecewise Compilation + Shared Graph Memory Pool (`Tier 4`, `vLLM V1` Paradigm)**: By compiling with symbolic shapes (`mark_dynamic`) once and splitting the FX graph at dynamic `Attention` ops (`splitting_ops`), each static compute piece (`Fused Add+RMSNorm+FP8`, `RoPE`, `SwiGLU`, GEMMs) is captured into bucketed CUDA Graphs that share a **single `torch.cuda.graph_pool_handle()` (`2.2 GB` total HBM, saving `16.2 GB`)**, achieving **`0` runtime recompilations** and **`6.8 ms` P99 TPOT (`1.62x` speedup)**!

---

## 1. 📌 Executive Summary

1. **Decouple `TorchInductor` Kernel Fusion from Monolithic `CUDAGraph` Capture (`Tier 3 & Tier 4`)**: `TorchInductor` generates symbolic Triton kernels that work for any sequence length `s0`, whereas `CUDAGraph` requires fixed memory addresses and static grid sizes.
2. **Use Regional Compilation (`TransformerBlock`-level) for Training (`Tier 3`)**: Compiling a single `TransformerBlock` with `mark_dynamic` reuses the compiled kernel across all 80 layers, cutting compile time by `7.7x` with `0` recompilations.
3. **Use Piecewise Compilation + Descending Shared-Pool Buckets for Serving (`Tier 4`)**: Exclude dynamic PagedAttention metadata ops from graph capture and record buckets from largest (`BS=512`) to smallest (`BS=1`) using one shared `graph_pool_handle()`.

---

## 2. 🎯 Core Architect Q&A (Q1 / Q2 / Q3)

### Q1: Why does `TorchDynamo` silently fall back to slow Eager execution after 8 shape changes (`Tier 1`)?
- **Answer**: When `cache_size_limit` (default `8`) is exceeded for a frame due to shape or scalar guard failures, Dynamo stops attempting new compilations and falls back to eager Python execution for unmatched shapes. Always use `mark_dynamic()` and set `torch._dynamo.config.error_on_recompile = True` after warmup.

### Q2: Why does capturing CUDA Graph buckets in descending order (`512 -> 256 -> ... -> 1`) with a shared `graph_pool_handle()` save `16.2 GB` of HBM?
- **Answer**: Because only one batch bucket executes at a time! When `BS=512` is captured first into `shared_pool`, it allocates the `2.2 GB` high-water mark. Every smaller bucket (`256, 128, ..., 1`) reuses those exact virtual addresses inside `shared_pool` with zero additional HBM allocation.

### Q3: Why must Attention / MLA ops be split out in `Piecewise Compilation` (`Tier 4`)?
- **Answer**: PagedAttention and FlashMLA kernels often depend on dynamic CPU-side metadata schedules (e.g., variable split-KV tile counts or prefix-cache lengths) that cannot be frozen into a single static CUDA Graph without massive padding waste. Splitting the FX graph around Attention gives both full CUDAGraph speed on linear/norm/activation layers and full flexibility on Attention.

---

## 3. 🧭 Visual Diagrams

```mermaid
flowchart LR
    A["Tier 1: Naive Full Compile (8 Recompiles -> Eager Fallback · 11.0 ms ✕)"] --> B["Tier 2: CUDAGraph Trees (+18.4 GB HBM Pool Bloat · 7.4 ms ⚠️)"]
    B --> C["Tier 4: Piecewise Compile + Shared 2.2 GB Pool (0 Recompiles · 6.8 ms 🏆)"]
```

---

## 4. 📊 4-Tier Architect Selection Matrix

| Metric | Tier 1: Naive Full Compile | Tier 2: `reduce-overhead` (`CUDAGraph Trees`) | Tier 3: Regional `mark_dynamic` | Tier 4: **Piecewise Compile + Shared Pool** |
| :--- | :--- | :--- | :--- | :--- |
| **Runtime Recompilations** | **`8` (Hits Limit -> Eager)** ✕ | `1` ⚠️ | **`0` (Symbolic `s0`)** ✅ | **`0` (Locked Post-Warmup)** 🏆 |
| **CUDA Graph Pool HBM** | `4.2 GB` ✕ | **`18.4 GB` (Disjoint Trees)** ✕ | **`0.0 GB`** ✅ | **`2.2 GB` (`-16.2 GB` Saved)** 🏆 |
| **P99 Decode `TPOT`** | `11.0 ms` (`1.00x`) ✕ | `7.4 ms` (`1.49x`) ⚠️ | `8.2 ms` (`1.34x`) ✅ | **`6.8 ms` (`1.62x` Speedup)** 🏆 |

---

## 5. 🧪 Quantitative Reference & Tooling

- **Evaluator & Sealed Data**: [`experiments/pytorch_compile_inductor_cudagraph_trees_dynamic_shape_selection/run.py`](../../experiments/pytorch_compile_inductor_cudagraph_trees_dynamic_shape_selection/run.py) · [`results.json`](../../experiments/pytorch_compile_inductor_cudagraph_trees_dynamic_shape_selection/results.json)
- **Interactive Sizer**: [`docs/reports/pytorch-compile-inductor-cudagraph-trees-dynamic-shape-recompilation-guard-sizer.html`](../reports/pytorch-compile-inductor-cudagraph-trees-dynamic-shape-recompilation-guard-sizer.html)
- **7-Card 4:5 Visual Report**: [`docs/reports/pytorch-compile-inductor-cudagraph-trees-dynamic-shape-recompilation-guard-selection-cards.html`](../reports/pytorch-compile-inductor-cudagraph-trees-dynamic-shape-recompilation-guard-selection-cards.html)

---

## 6. 🛠️ Good Fit vs. Not For & Production Checklist

- **✅ Good fit**: Piecewise compilation with shared-pool CUDA Graph buckets for LLM serving (`vLLM V1`), and Regional `TransformerBlock` compilation with `mark_dynamic` for training.
- **⛔ Not for**: Naive full-model `mode="reduce-overhead"` on variable-length PagedAttention workloads.
- [ ] Enable `TORCH_LOGS="recompiles"` during staging and set `error_on_recompile = True` after warmup.
- [ ] Capture CUDA Graph buckets in descending batch-size order using a single shared `graph_pool_handle()`.
