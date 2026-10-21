# FlashInfer TRTLLM-Gen Sliding/Variable-Window FMHA & Cake MiniMax-H3 / Small & Dense-Projection GEMM Consolidation

> **Primary Upstream Commits**:
> - [`flashinfer-ai/flashinfer@d438333`](https://github.com/flashinfer-ai/flashinfer/commit/d4383335bae57d307edfcb58dce07bc78af2dece) (`#4905`, 2026-10-21) — *feat: integrate TRTLLM-Gen sliding window and variable window FMHA*
> - [`flashinfer-ai/flashinfer@c785284`](https://github.com/flashinfer-ai/flashinfer/commit/c78528451fcd327e6f81fb5586445106cd11abe3) (`#6947`, 2026-10-21) — *[Cake] MiniMax-H3 SM100/SM103 round-2 consolidation*
> - [`flashinfer-ai/flashinfer@a509a27`](https://github.com/flashinfer-ai/flashinfer/commit/a509a27181cbbc22887384c8780cbd5e6bc7b7ca) (`#6946`, 2026-10-21) — *[Cake] Consolidate small GEMMs (`router_gemm`, `bf16_bmm`, `quant_lora_gemm`)*
> - [`flashinfer-ai/flashinfer@2852797`](https://github.com/flashinfer-ai/flashinfer/commit/28527974293499a21687feccc2c6ebf1009d1) (`#6938`, 2026-10-21) — *[Cake] Consolidate `dense_projection_gemm` across SM100/SM107*

---

## 1. Background & Core Bottlenecks

On Blackwell (SM100a / SM103a / SM107a) inference pipelines, long-context attention and the Cake AoT kernel library faced two architectural challenges:
1. **Per-Query-Token Variable-Window & Bi-Directional Sliding-Window FMHA**: Hybrid models (Gemma-3, MiniMax-H3, sparse block attention) require per-query-token inclusive KV bounds $[s_i, e_i]$ (`VariableWindow`) or bi-directional windows $(W_{\text{left}}, W_{\text{right}})$, while short-sequence decode steps ($\text{max\_kv\_len} \le W_{\text{left}} + 1$) must preserve the zero-overhead `Causal` fast path.
2. **Cake AoT Source Duplication**: Across `minimax_h3`, `cake_router_gemm`, `cake_bf16_bmm`, `cake_quant_lora_gemm`, and `dense_projection_gemm`, per-tile and per-cluster source clones produced over 45,000 duplicated lines and dead `#if __CUDA_ARCH__ == 1000` branches in SM107-only translation units.

---

## 2. TRTLLM-Gen Sliding & VariableWindow FMHA Mechanics (`#4905`)

### 2.1 Six-Mode Mask Dispatch & CUBIN Artifact Expansion
`flashinfer@d438333` extends `TrtllmGenAttentionMaskType` to 6 modes:
- `Dense = 0`, `Causal = 1`, `SlidingOrChunkedCausal = 2`, `Custom = 3`, `SlidingWindowCustom = 4` (`window_right > 0`), and `VariableWindow = 5` (per-query-token inclusive `variable_window_token_starts` / `variable_window_token_ends`).
- Updates the FMHA CUBIN artifact to `35cb99413a3ebce2e87a03fc04196e528e22c8d5` (`36,964` total FMHA CUBINs, including **`1,424` `VariableWindow` CUBINs**).

### 2.2 Non-Binding Window Causal Fast Path & Bug Fixes
- **`fixedWindowBinds` Fast Path**: When `window_left >= 0` but `max_kv_len <= window_left + 1`, `fixedWindowBinds` evaluates to `false`, preserving `TrtllmGenAttentionMaskType::Causal`. On **B300** (`B=8, KV=128, Hq=16, Hkv=4, D=128, P=16`), unbounded decode and non-binding `window_left=127` both measure **`4.2624 us`** (vs **`4.4736 us`** when binding at `KV=4096`).
- **VariableWindow Prefill Overhead**: At `B=8, Q=512, KV=4096`, Fixed Window vs VariableWindow measures **`29.438 -> 30.667 us` (`+4.17%`)** on **B300** and **`35.520 -> 36.831 us` (`+3.69%`)** on **B200**.
- **Two Correctness Fixes**:
  1. Removes the trailing `runnerParams.mChunkedAttentionSizeLog2 = 0;` overwrite that previously clobbered enabled chunked attention;
  2. Rejects `window_left >= 0` with `causal=False` in `BatchPrefillWithPagedKVCacheWrapper.plan()`.

---

## 3. Cake Three-Wave Consolidation (`-39,298` Net Lines, 100% SASS-Identical)

| PR / Commit | Target Family | Consolidation Mechanics | Net Line Delta | Binary / Numerical Equivalence |
|---|---|---|---|---|
| **`#6947`** (`c785284`) | `minimax_h3` (SM100/SM103) | Drops 30 duplicate vendored device-helper headers (`904` lines) & 112 unused macros (`224` lines); extracts 3 shared `norm+AdaLN` headers (`bf16/mxfp8/nvfp4`, removing `733` duplicated lines) | `+829 / -2,598` (**`-1,769`** net, 39 files) | **`65/65`** CUBIN SHA-256 & SASS identical; **`77/77`** bit-exact GPU cells on B200/GB300 |
| **`#6946`** (`a509a27`) | `router_gemm` / `bf16_bmm` / `quant_lora_gemm` | Folds 32 `router_gemm` sources onto 1 body, 32/34 `bf16_bmm` sources onto 6 bodies, 2 `quant_lora_gemm` hint kernels onto partner sources, and shares 3 host bindings across 9 LoRA programs | `+2,759 / -19,641` (**`-16,882`** net, 89 files) | **`143/143`** + **`21/21`** SASS-identical device rows |
| **`#6938`** (`2852797`) | `dense_projection_gemm` (SM100/SM107) | Folds 81 duplicate host bindings into 6 shared shape launchers, folds 10 clone kernels into 7 bodies, drops dead `SM100` arms from 27 SM107-only kernels, derives `44 + 54 - 14 = 84` `EXPORTED_TEMPLATES` | `+2,650 / -23,297` (**`-20,647`** net, 135 files) | 87 programs / **104 builds** (47 SM100 + 57 SM107) 100% SASS-identical |

---

## 4. Deterministic Reproduction

```bash
python experiments/flashinfer-trtllm-gen-sliding-varwindow-fmha-cake-minimax-h3-small-dense-gemm/run.py --check
```
