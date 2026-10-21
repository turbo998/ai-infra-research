# Deep Dive into FlashInfer Unified `frost_low_latency` MoE (`SM100/SM103/SM107`), `cake_rmsnorm_train` Row-Window Tree `dweight`, `cake_sage` One-Step-Ahead Softmax Metadata, `sm_107a` Look-Back SSD & `sm_90a` Compact Static VSA

> **Language / 语言**: [English](flashinfer-frost-low-latency-moe-cake-rmsnorm-sage-vsa.en.md) | [中文](flashinfer-frost-low-latency-moe-cake-rmsnorm-sage-vsa.md)  
> **Phase**: Phase 49 (`priority: 58`) · **Date**: 2026-10-22  
> **Primary Upstream Sources**:
> - `flashinfer-ai/flashinfer@2995461e872e030b364a1a84d8ca94181f857876` (`#6626`, 2026-10-21)
> - `flashinfer-ai/flashinfer@d892cd4197149d60c0e86ca0026a3df9dfe111fb` (`#6939`, 2026-10-21)
> - `flashinfer-ai/flashinfer@c21f621ae75ed3cf28cae7745281533e1bf78065` (`#6951`, 2026-10-21)
> - `flashinfer-ai/flashinfer@7601d0bb7e92390aae5276951f0e4b7c8a2b5c14` (`#6943`, 2026-10-21)
> - `flashinfer-ai/flashinfer@7808983872ee3961f142bce3227d7ecc2afa7253` (`#6949`, 2026-10-21)

---

## 1. Context & Problem Statement

Across Blackwell (`sm_100a` B200 / `sm_103a` B300/GB300), Rubin (`sm_107a` R200), and Hopper (`sm_90a` H100), FlashInfer landed five architectural kernel upgrades targeting low-latency MoE decode, deterministic training RMSNorm backward reductions, FP8 block-sparse attention, Mamba2 SSD state scans, and Hopper variable block-sparse attention (VSA):

1. **Unified `frost_low_latency` MoE Backend (`#6626`)**: Previously split into separate `frost_bf16_moe` and `frost_mxfp4_moe` backends with incompatible prepared-weight layouts between Tensor Core and FMA tactics, limited activation support, and a hard `top-k <= 8` ceiling on FMA.
2. **`cake_rmsnorm_train` Row-Count Window Backward & Fixed-Order Tree Reduction (`#6939`)**: Multi-CTA backward reductions for `dweight` need bitwise determinism without launching separate zero-fill kernels for barrier counters.
3. **`cake_sage` One-Step-Ahead Softmax Metadata Staging (`#6951`)**: The `SM100/SM103` Sage-FP8 block-sparse attention kernel previously stalled at each softmax step waiting on dependent global loads of block metadata and per-16-token `K` scales while two warps remained idle.
4. **`cake_ssd_combined` `sm_107a` Look-Back Chunk-Parallel SSD (`#6943`)**: Optimizing the 6 look-back chunk-parallel Mamba2 SSD programs (`bf16/f16/f32 × batched/varlen`) on Rubin (`sm_107a`).
5. **`cake_vsa_sm90` Persistent Kernel & Compact Static Program (`#6949`)**: Adding a compact static persistent program for short tiles with high tiles-per-CTA alongside 17 refreshed Hopper (`sm_90a`) VSA programs.

---

## 2. Architectural & Kernel Mechanism Analysis

### 2.1 Unified `frost_low_latency` MoE (`#6626`)
- **`FrostLowLatencyConfig` & Auto-Routing**: Replaces `frost_bf16_moe` / `frost_mxfp4_moe` with `frost_low_latency` (`default max_num_tokens=2`, tunable across `1 <= max_num_tokens <= 8`), automatically selected when `backend="auto"` and `1 <= num_tokens <= max_num_tokens`.
- **Shared Prepared Weights & 18-Tactic Autotuning Space**:
  - On `SM100/SM103/SM107`, MXFP4 and MXFP8 Tensor Core tactics (`TC128` and `TC256` with Programmatic Dependent Launch, PDL) and scalar FMA tactics (`top-k <= 32`) **share the exact same prepared weight layout**. A single `AutoTuner` pass sweeps both TC tactics plus all `16` FC1/FC2 warp-output-width combinations (`1, 2, 4, 8 × 1, 2, 4, 8`).
  - BF16 FMA supports **10 typed activations** (`SwiGLU`, `SiTU`, `GeGLU`, `GeGLUTanh`, `SwiGLUStep`, `GELU`, `ReLU`, `ReLU2`, `SiLU`, `Identity`).
- **Measured NVIDIA B300 (`sm_103a`) Speedups vs CUTLASS**:
  - **BF16**: **`1.794×`** (`27.736 -> 15.460 µs` on `T=1, SwiGLU`), **`1.385×`** (`44.144 -> 31.872 µs` on `T=2, SwiGLU`), **`1.312×`** (`60.424 -> 46.056 µs` on `T=2, SiTU`).
  - **MX Clamped-SwiGLU**: **`1.608×`** (`21.616 -> 13.440 µs` on `W4/T1`), **`1.472×`** (`31.852 -> 21.632 µs` on `W4/T4`), **`1.257×`** (`40.008 -> 31.816 µs` on `W8/T4`).

### 2.2 `cake_rmsnorm_train` Row-Window Backward & Fixed-Order Tree Reduction (`#6939`)
- **Row-Count Window Backward**: On `sm_100a`, `sm_103a`, and `sm_107a` (`H=1536, 4096`), each persistent CTA computes `dx` (or fused `dx, du`) over a contiguous row window and writes its partial `dweight` row into `partial[chunk_id]`.
- **Self-Resetting Flip Barrier & Tree Reduction**: CTAs synchronize via a self-resetting flip barrier (`counters` resets to zero on the last arriving CTA and flips its sense bit, avoiding a separate zero-fill launch) and reduce `partial` into `dw` using a **fixed-order tree reduction** independent of CTA arrival order.
- **Measured Speedups**: Up to **`1.079×`** forward (`sm_107a` `H=1536` bulk-copy producer/consumer above `T=12288`) and **`1.134×`** backward on R200 (`sm_107a`), **`1.118×`** backward on B200 (`sm_100a`), and **`1.075×`** backward on GB300 (`sm_103a`).

### 2.3 `cake_sage`, `cake_ssd_combined` & `cake_vsa_sm90` (`#6951`, `#6943`, `#6949`)
- **`cake_sage` One-Step-Ahead Softmax Metadata (`#6951`)**: Two previously idle warps stage block metadata and per-16-token `K` scales into shared memory one step ahead (`dynamic_smem_bytes` `119040 -> 120320 B`, `+1280 B`), achieving bit-identical `O`/`LSE` with **`1.08×`** speedup on B200 (`0.3552 -> 0.3288 ms` on `A1`) and **`1.03×`** on GB300 (`0.2922 -> 0.2824 ms`).
- **`cake_ssd_combined` `sm_107a` Look-Back SSD (`#6943`)**: Refreshes all 6 `sm_107a` look-back chunk-parallel programs on R200, reaching **`1.0215×`** geomean call-level speedup (peak **`1.196×`**, `92.1 -> 77.0 µs` on `perf_batched_1x256`) and **`1.0839×`** geomean main-kernel speedup (peak **`1.268×`**, `33.3 -> 26.2 µs` on `cp_b4h8c16`).
- **`cake_vsa_sm90` 18-Program Suite (`#6949`)**: Delivers 3 persistent programs (including the new compact static program for uniform short-tile plans) and 15 small-selection variants on H100 SXM (`sm_90a`), achieving **`1.2624×`** (`13.472 -> 10.680 µs` on `h1-m512-n8192-k48`) and **`1.2237×`** (`35.360 -> 28.912 µs` on `h8-m128-n32768-k64`).

---

## 3. Deterministic Standard-Library Verification

Verified via `experiments/flashinfer-frost-low-latency-moe-cake-rmsnorm-sage-vsa/run.py --check`:

| Dimension | Invariant / Metric | Reproduced Value | Status |
|---|---|---|---|
| `frost_low_latency` Tactic Space | Typed Activations / FC1×FC2 Warp Pairs | `10` activations / `16` pairs (`18` MX tactics) | `PASS` |
| `frost_low_latency` B300 Speedups | BF16 (`T=1` SwiGLU) / MX (`W4/T1`) | **`1.794×`** (`27.736→15.460 µs`) / **`1.608×`** (`21.616→13.440 µs`) | `PASS` |
| `cake_rmsnorm_train` Determinism | Arrival-Order Invariant Tree `dw` & Flip Barrier | `tree_bitwise_deterministic = True`, R200 bwd **`1.134×`** | `PASS` |
| `cake_sage` Shared Memory Staging | `dynamic_smem_bytes` Delta / B200 `A1` Speedup | `119040 -> 120320 B` (`+1280 B`), B200 **`1.08×`** | `PASS` |
| `cake_ssd_combined` & `cake_vsa_sm90` | R200 SSD Peak / H100 VSA 18-Program Peak | SSD **`1.268×`** (`1.196×` call) / VSA **`1.2624×`** | `PASS` |
