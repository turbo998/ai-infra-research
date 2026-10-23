# Deep Dive: FlashInfer Cake GDP Two-Head T=1 Decode & Split-V Reduce-Scatter, BF16 Cold-Chain Autotuner L2 Policy, Frost Single-Entry TVM-FFI, and TileLang Step/AtomicType Fixes

> **Language / 语言**: [English](flashinfer-cake-gdp-2head-cold-chain-frost-ffi.en.md) | [中文](flashinfer-cake-gdp-2head-cold-chain-frost-ffi.md)
> **Question ID**: `flashinfer-cake-gdp-2head-cold-chain-frost-ffi` (`priority: 76`, Phase 67 Track A)
> **Evidence Class**: `implementation` + `author-benchmark` + `own-offline`

---

## 1. Problem Statement & Architectural Context

Across Blackwell (`sm_100a` / `sm_103a`) kernel dispatch and compiler code generation, four subtle bottlenecks degrade single-token (`T=1`) decode latency and correctness:

1. **Cake Gated DeltaProduct (`cake_gated_delta_product_mtp`) Two-Regime `T=1` Bottlenecks**:
   - In the many-wave regime (`B * HV >= 32 * sm_count`), the single-head-per-CTA kernel (`l8r4`, `6d077dd3b5a74469db0d`) launches one CTA per `(batch, head)` row, incurring excessive CTA scheduling overhead.
   - In the small-batch `T=1` regime (`B * HV <= 160`), the `splitv8_tile8` kernel reduced two rows (`w` and `w+4`) per warp via two separate 5-step full-warp `__shfl_xor_sync(16..1)` butterfly trees (10 shuffles + 2 `sV` shared loads per substep).
2. **Single-Launch Floor Noise in Blackwell Low-M (`M <= 64`) BF16 GEMM Autotuning**:
   - On B300, a single isolated CUDA-graph launch has a `~6.38 us` launch floor (`aten::add_` = `6.38 us`) plus kernel-family-dependent first-launch overhead (`cluster split-K > warp split-K > direct`). Candidates whose true steady-state latency is `2.8–6.4 us` compress around `~10.24 us` under single-launch (`L=1`) timing, causing the autotuner to pick slower tactics.
3. **Frost Low-Latency MoE Host Dispatch Overhead**:
   - Every Frost low-latency MoE call executes 2 (FMA) or 4 (MX TC) kernels. Previously, `@cute.jit` wrapped individual kernels, forcing Python to reconstruct `from_dlpack` and raw pointer descriptors on every launch (`106–261 us` CPU time vs `10–27 us` GPU time at `T=1`).
4. **TileLang Non-Unit Loop Vectorization & Unsigned Atomic Type Normalization Bugs**:
   - `LoopVectorizer` ignored `op->step` when passing `(loop_var, extent, body)` to `TLVectorizer::Vectorize`, silently turning stepped loops into unit-stride loops. Meanwhile, CUDA codegen cast `AtomicMax/Min/AddRet/Store` pointers to `normalize_atomic_type<T1>::type` (`NT1`), turning signed `int64` into `unsigned long long` and breaking `bf16`/`fp16` `atomic_store`.

---

## 2. Upstream Implementation & Mechanism Analysis

### 2.1 FlashInfer Cake GDP: Two-Head/CTA (`l8r4x2`) & Split-V Reduce-Scatter (`#7128`)

In `flashinfer-ai/flashinfer@58232f576ad772c7c7f10315073c306f9cf0a292`:

- **Two-Head/CTA `T=1` Program (`cake_gdp_266d24496704b9063e28`, `l8r4x2`)**:
  - Dispatched when `T == 1`, architecture is `sm_100a`/`sm_103a`, and `B * HV >= 32 * sm_count`.
  - Packs two `(batch, head)` rows into each CTA with grid size `(B * HV + 1) // 2`, safely handling odd `B * HV` (e.g., `32 * sm_count + 1` or `1713 * 3 = 5139`) where the final CTA processes a single head.
- **Split-V (`splitv8_tile8`, `B * HV <= 160`) Reduce-Scatter + All-Gather (`cake_gdp_f2f0c7abbf10082a6996`)**:
  - Assigns half-warp row ownership via `row_own = (lane_local >> 4) & 1` so lanes `0..15` own `row_a = w` (`row_b = w+4`) and lanes `16..31` own `row_a = w+4` (`row_b = w`).
  - Executes **1 cross-half exchange** `sum_a += __shfl_xor_sync(sum_b, 16)`, **4 intra-half shuffles** (`offset = 8, 4, 2, 1`), a single `sV` load per half-warp (`vn_a = (v_a - sum_a) * beta`), and **1 all-gather** `vn_b = __shfl_xor_sync(vn_a, 16)`.
  - Cuts warp shuffles per substep from **10 to 6 (-40.0%)** and readout shuffles from **10 to 5 (-50.0%)**.
- **Author Benchmarks (`author-benchmark`, 48 shapes)**:
  - **B200**: `0.98x–1.18x` (`HV=80, B=256, T=1`: `243.4 -> 206.1 us`, **`1.18x`**; `HV=10, B=16, T=1`: `5.8 -> 5.3 us`, **`1.09x`**).
  - **GB300**: `0.99x–1.13x` (`HV=80, B=256, T=1`: `228.0 -> 201.7 us`, **`1.13x`**; `HV=80, B=64, T=1`: `62.0 -> 57.2 us`, **`1.08x`**).

### 2.2 FlashInfer BF16 GEMM: `cold-chain:8` Autotuner L2 Policy (`#6360`)

In `flashinfer-ai/flashinfer@aec44bda63e69b657b7fee4935a3d5b2ffe08462`:

- Adds `cold_chain_launches: int = 8` on `_BF16_GEMM_SM100_TUNING_CONFIG` (`l2_cache_policy="cold-chain:8"`).
- Captures `L=8` back-to-back launches across distinct `ValueProfileArena` slots after an L2 flush and divides the elapsed graph time by `8`, amortizing the first-launch penalty while keeping every launch L2-cold.
- At `M=16, N=512, K=8192` (`pdl=1`), isolated `L=1` timing inverted `SplitK(64,8,4,12)` (steady `3.74 us`, `L=1` `12.29 us`) vs `WarpSplitK(16,8,256,16,1)` (steady `5.17 us`, `L=1` `10.34 us`). Under `cold-chain:8`, `SplitK` measures `4.03 us` vs `WarpSplitK` `4.48 us`, selecting the true best tactic.
- Caps samples per candidate to `~10 ms` of chain time (`10 / 2 / 1` samples for `0.01 / 4.0 / 50.0 ms` chains) and uses `_raw_storage_span_elems(t)` so non-contiguous/padded tensors never overlap across arena slots.
- **Author Benchmarks (`author-benchmark`, B300 60 low-M cells)**: Tuned vs hand geomean improves from `1.017x` (`17/60` cells losing `>5%`, worst `0.719x`) to **`1.124x`** (**`0/60`** cells losing `>5%`, worst `0.983x`); `backend="auto"` (`pdl=1`, 30 cells) drops `>5%` misses from `21/30` to **`1/30`** (geomean overhead `1.171x -> 1.006x`).

### 2.3 FlashInfer Frost Single-Entry TVM-FFI (`#7127`) & TileLang Fixes (`#3345`, `#3307`)

1. **Frost Low-Latency MoE Single-Entry TVM-FFI (`c0e51ae29b19960037ae064b0ec0c45b636a312d`)**:
   - Compiles one `@cute.jit` TVM-FFI `launch` function per tactic (`bf16-ffi-v1`, `nvfp4-ffi-v1`, `mx-ffma-ffi-v1`, `mx-tc-ffi-v1`) that chains all 2 or 4 sub-kernels in compiled host code.
   - On B300 (`H=2048, I=1408, E=64, top_k=6, T=1 SwiGLU`), warm public API CPU time speeds up **`2.85x–4.65x`** (`MXFP4 TC128`: `260.85 -> 61.85 us`, `4.22x`; prepacked runner: `217.82 -> 24.90 us`, `8.75x`) with unchanged GPU time (`10.33 us`).
2. **TileLang Step Guard & CUDA AtomicType Preservation (`cb26ee0fafe5`, `2573d470798a`)**:
   - `LoopVectorizer` checks `ICHECK(!op->step.has_value() || is_one(op->step.value()))` so stepped loops are never vectorized with a lost step.
   - `AtomicMax`, `AtomicMin`, `AtomicAddRet`, and `AtomicStore` use `cuda::atomic_ref<T1, cuda::thread_scope_device>` directly instead of unsigned `NT1`, preserving signed `int64` ordering and `bf16`/`fp16` atomic stores.

---

## 3. Deterministic Verification Summary (`own-offline`)

Verified by `python experiments/flashinfer-cake-gdp-2head-cold-chain-frost-ffi/run.py --check`:

| Metric / Invariant | Baseline / Before | Optimized / After | Deterministic Finding |
|---|---|---|---|
| Cake GDP Split-V shuffles per substep | `10` `__shfl_xor_sync` | `6` `__shfl_xor_sync` | **`-40.0%` shuffles**, exact 32-lane match (`< 1e-9`) |
| Cake GDP `HV=80, B=256, T=1` (B200 / GB300) | `243.4 us` / `228.0 us` | `206.1 us` / `201.7 us` | **`1.18x` / `1.13x`** (`l8r4x2` two-head CTA) |
| B300 `M=16, N=512, K=8192` autotuner pick | `L=1` picks `WarpSplitK` (`5.17 us`) | `cold-chain:8` picks `SplitK` (`3.74 us`) | `>5%` misses drop from `21/30` to **`1/30`** |
| Frost `T=1` `MXFP4 TC128` CPU time (Public / Runner) | `260.85 us` / `217.82 us` | `61.85 us` / `24.90 us` | **`4.22x` / `8.75x`**, GPU unchanged at `10.33 us` |
| TileLang `int64` `atomic_max(-(1<<40), 1<<40)` | `NT1 (uint64)` returns `-1099511627776` | `T1 (int64)` returns `+1099511627776` | Signed `int64` ordering & step guard restored |
