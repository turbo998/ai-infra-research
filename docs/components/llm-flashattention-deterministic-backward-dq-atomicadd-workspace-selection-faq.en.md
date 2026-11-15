# Why `FlashAttention-2/3/4` Backward Pass Is Non-Deterministic by Default (`dQ_accum` `atomicAdd` Race Across KV CTAs) & Bitwise-Deterministic (`deterministic=True`) Selection Guide

> **Provenance & Evidence Tier**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`). Grounded in [`experiments/llm_flashattention_deterministic_backward_splitkv_atomic_workspace_selection/results.json`](../../experiments/llm_flashattention_deterministic_backward_splitkv_atomic_workspace_selection/results.json).
> **中文版本**: [`FlashAttention-2/3/4` 反向传播为什么默认不具确定性（跨 CTA 并行切分 `K/V` 对 `dQ_accum` 执行乱序 `atomicAdd`）与逐比特可复现训练（`deterministic=True`）选型指南](./llm-flashattention-deterministic-backward-dq-atomicadd-workspace-selection-faq.md)

![FlashAttention Backward Non-Determinism & Ordered dQ Accumulation Summary](../../assets/svg/llm-flashattention-deterministic-backward-dq-atomicadd-workspace-selection-faq-summary.svg)

---

## 0. 🔥 Daily Hottest Capability Breakthrough

- **Achieving `0 ULP` Bitwise-Reproducible `FlashAttention-2.6+ / FA3` Backward Passes at `785.0 TFLOPS` (Only `4.2%` Overhead vs. `820.0 TFLOPS` Non-Deterministic Default, and `2.53x` Faster Than Serial Fallback) via SM-Bounded Split Workspaces and Ordered Semaphores**: Why does running the exact same LLM training step twice with identical random seeds produce different gradients?
  1. **The Root Cause: Concurrent `atomicAdd` into Global FP32 `dQ_accum` Across KV CTAs (`Tier 1`)**: While the `FlashAttention` forward pass parallelizes across `Q` blocks (so each CTA writes a disjoint slice of output `O` deterministically), the **`FlashAttention-2 / FA3` backward pass parallelizes CTAs across the `K/V` sequence dimension**! Each CTA holds a block of `K_j, V_j` in SRAM, loops over `Q_i` blocks, and accumulates its partial `dQ_i` contribution into a shared global HBM FP32 buffer (`dQ_accum`) using hardware **`atomicAdd()`**! Because SM scheduling order across CTAs varies from run to run and floating-point addition is non-associative, `dQ` drifts by **`~6 ULPs` per step**, causing training trajectories to diverge!
  2. **Serial Single-CTA Fallback (`Tier 2` — `310.0 TFLOPS`) & Unbounded Split Buffers (`Tier 3` — `+16.0 GB` HBM)**: Disabling KV-block parallelism leaves SMs idle under GQA/MLA (`-62.2%` throughput), while allocating `[NumSplits, B, H, S, D]` FP32 buffers wastes `16.0 GB` of HBM on `32K` sequences.
  3. **`FA2.6+ / FA3` Ordered `deterministic=True` (`Tier 4`)**: Retains full KV-dimension CTA parallelism (`785.0 TFLOPS`) while using **SM-bounded wave workspaces (`0.50 GB`) or hardware ordered semaphores (`tile_count_semaphore`)** to force `dQ_accum` additions to occur in strict KV-block order (`j = 0, 1, 2, ...`), guaranteeing **`0 ULP` bitwise reproducibility** with just **`4.2%` overhead**!

---

## 1. 📌 Executive Summary

1. **Forward Is Deterministic by Default; Backward Is Not (`Tier 1`)**: `FlashAttention-2/3` backward uses cross-CTA hardware `atomicAdd` on `dQ_accum`, introducing ~`6 ULP` run-to-run jitter.
2. **Never Disable KV Parallelism to Get Determinism (`Tier 2`)**: On GQA (`num_kv_heads = 8`) or MLA (`num_heads = 1`), serializing the KV loop starves Hopper/Blackwell's 132+ SMs (`310 TFLOPS`).
3. **Pass `deterministic=True` for SDC Auditing, Loss-Spike Replay, and CI Verification (`Tier 4`)**: Delivers `0 ULP` bitwise determinism at `785.0 TFLOPS` (`95.7%` of peak speed) with only `0.50 GB` workspace.

---

## 2. 🎯 Core Architect Q&A (Q1 / Q2 / Q3)

### Q1: How does `FlashAttention-3` (`deterministic=True`) enforce deterministic `dQ_accum` addition without allocating a `16 GB` buffer?
- **Answer**: Even while all CTAs compute their `WGMMA` matrix multiplications for `dK`, `dV`, and local `dQ` in parallel inside on-chip SRAM/registers, they coordinate the final global write-back to `dQ_accum[i]` using a lightweight global memory barrier/semaphore array indexed by query tile `i` (or a wave-bounded split buffer sized to `num_SMs`), ensuring that KV block `j` always adds its partial `dQ` strictly after KV block `j - 1`!

### Q2: Why does non-deterministic `atomicAdd` matter if `6 ULPs` of zero-mean noise doesn't hurt final model accuracy?
- **Answer**: Without bitwise reproducibility (`0 ULP`), you **cannot**:
  1. Run online **Silent Data Corruption (SDC)** golden-step verification (where a healthy shadow check expects an exact bitwise match to prove the GPU ALU has no hardware bit flips);
  2. Deterministically replay a checkpoint from 10 steps before a **Loss Spike / NaN crash**;
  3. Verify that a communication or compiler optimization (`FSDP2`, `ZigZag CP`) is mathematically bug-free.

### Q3: What other flags must be paired with `flash_attn_func(..., deterministic=True)` for end-to-end bitwise reproducibility in PyTorch?
- **Answer**: Set `torch.use_deterministic_algorithms(True)` and `export CUBLAS_WORKSPACE_CONFIG=:4096:8` (which prevents cuBLAS backward GEMMs from using non-deterministic `Split-K atomicAdd` reductions).

---

## 3. 🧭 Visual Diagrams

```mermaid
flowchart LR
    A["Tier 1: Default Unordered atomicAdd (820 TFLOPS · 6-ULP Drift ✕)"] --> B["Tier 2: Serial 1-CTA Fallback (310 TFLOPS · 0 ULP ⚠️)"]
    B --> C["Tier 4: FA2.6+/FA3 Ordered deterministic=True (785 TFLOPS · 0 ULP · 0.5GB 🏆)"]
```

---

## 4. 📊 4-Tier Architect Selection Matrix

| Metric | Tier 1: Default `atomicAdd` (`False`) | Tier 2: Serial 1-CTA Fallback | Tier 3: Unbounded Split Buffer | Tier 4: **Ordered `deterministic=True`** |
| :--- | :--- | :--- | :--- | :--- |
| **Max Per-Step Drift (`ULP`)** | **`6 ULPs` (Diverges)** ✕ | `0 ULP` (Bitwise Exact) ✅ | `0 ULP` (Bitwise Exact) ✅ | **`0 ULP` (100% Bitwise Exact)** 🏆 |
| **Extra `dQ` Workspace HBM** | `0.25 GB` ✅ | `0.25 GB` ✅ | **`16.00 GB` (HBM Bloat)** ✕ | **`0.50 GB` (SM-Bounded)** 🏆 |
| **Backward Throughput** | `820.0 TFLOPS` (`100%`) | **`310.0 TFLOPS` (`-62.2%`)** ✕ | `710.0 TFLOPS` ⚠️ | **`785.0 TFLOPS` (`95.7%` Peak, `2.53x`)** 🏆 |

---

## 5. 🧪 Quantitative Reference & Tooling

- **Evaluator & Sealed Data**: [`experiments/llm_flashattention_deterministic_backward_splitkv_atomic_workspace_selection/run.py`](../../experiments/llm_flashattention_deterministic_backward_splitkv_atomic_workspace_selection/run.py) · [`results.json`](../../experiments/llm_flashattention_deterministic_backward_splitkv_atomic_workspace_selection/results.json)
- **Interactive Sizer**: [`docs/reports/llm-flashattention-deterministic-backward-dq-atomicadd-workspace-sizer.html`](../reports/llm-flashattention-deterministic-backward-dq-atomicadd-workspace-sizer.html)
- **7-Card 4:5 Visual Report**: [`docs/reports/llm-flashattention-deterministic-backward-dq-atomicadd-workspace-selection-cards.html`](../reports/llm-flashattention-deterministic-backward-dq-atomicadd-workspace-selection-cards.html)

---

## 6. 🛠️ Good Fit vs. Not For & Production Checklist

- **✅ Good fit**: Enabling `deterministic=True` during periodic SDC golden-step audits, loss-spike debugging replays, and deterministic RL / CI equivalence tests.
- **⛔ Not for**: Using serial 1-CTA attention fallbacks (`Tier 2`) or expecting bitwise identical gradients when `deterministic=False`.
- [ ] Pass `deterministic=True` to `flash_attn_func` / `flash_attn_varlen_func` during SDC verification and debugging replays.
- [ ] Pair with `CUBLAS_WORKSPACE_CONFIG=:4096:8` and `torch.use_deterministic_algorithms(True)`.
