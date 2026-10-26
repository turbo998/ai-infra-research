# FlashInfer Cake SM90 INT8/FP8 4-Stage Attention, `cake_tinygemm2_tcgen05` Narrow-Tile Unrolling, `cake_fp4_allreduce` Registry Consolidation & TileLang SIMT Vector-Local Offset Analysis (Phase 100 · 100-Phase Milestone)

- **Date / Phase**: 2026-10-27 · Phase 100 (`priority: 109` — **100-Phase Milestone!**)
- **Component**: `gpu-compute-architecture`
- **Question ID**: `q-flashinfer-cake-fp8-attention-tcgen05-narrow-unroll-fp4-allreduce-phase100`
- **Evidence ID**: `claim-flashinfer-cake-fp8-sm90-tcgen05-narrow-unroll-phase100`
- **Reproducible Experiment Suite**: [`experiments/flashinfer_cake_fp8_tcgen05_phase100/`](../../experiments/flashinfer_cake_fp8_tcgen05_phase100/run.py)

---

## 1. Architectural Context & Core Problem

Across Hopper (`sm_90a`) and Blackwell (`sm_100a` / `sm_103a` / `sm_107a` / `sm_120a`) kernel stacks, four engineering bottlenecks arise:

1. **Dynamic Smoothing & Quantization Overhead in Low-Bit Dense MHA**: Running INT8 QK + FP8 E4M3 PV attention with BF16 inputs/outputs requires on-GPU statistics reduction, K-mean smoothing, and quantization on every call. Below a critical sequence length, the 4-kernel chain overhead outweighs the Tensor Core compute gain relative to BF16 FlashAttention-3.
2. **Loop-Control Overhead in Blackwell `tcgen05` Narrow Tiles**: In `flashinfer.gemm.tinygemm_bf16`, single-round narrow-tile MMA/TMA loops suffer from per-step K predicate checks and loop control overhead on small-batch long-$K$ projections.
3. **Exporter Code Bloat in `cake_fp4_allreduce`**: The NVFP4-transport fused RMSNorm all-reduce package accumulated 116 `.cu` files (252,517 lines), including unreachable prefetch (`_pf`) programs requiring $\ge 212$ CTAs and 53 duplicated TVM-FFI host bindings.
4. **CUDA Graph Schedule Lifetime across Provider Switches & SIMT Vector-Local Indexing**: Switching `auto` prefill providers after capturing FA2 leaves FA2's CPU-generated output/merge schedule stale, while TileLang's Ascend PTO codegen allocated scalar local buffers for vector slots without multiplying by `element_lanes`.

---

## 2. Primary Source Empirical Analysis (10 Zero-Overlap Commits)

### 2.1 FlashInfer `#7373` (`14165a5bd9`): SM90 Cake INT8 QK + FP8 E4M3 PV 4-Kernel Chain

`flashinfer/experimental/cake_fp8_attention` exposes `prepare_fp8_attention(q, k, v, out, backend="cake")` and `fp8_attention_workspace_bytes` for dense non-causal MHA ($D=128$, BF16 in/out, FP32 softmax, arbitrary NHD/HND views) on `sm_90a`. Every call executes a 4-kernel chain under CUDA-graph replay:

$$\text{BF16 } (Q,K,V) \xrightarrow{\text{1. stats\_partials}} \text{Partials} \xrightarrow{\text{2. stats\_finalize}} (\mu_K, s_Q, s_K, s_V) \xrightarrow{\text{3. quantize}} (\hat{Q}_{\text{INT8}}, \hat{K}_{\text{INT8}}, \hat{V}_{\text{E4M3}}) \xrightarrow{\text{4. attention}} \text{BF16 Out}$$

Author-reported end-to-end benchmarks on NVIDIA H100 80GB HBM3 (`sm_90a`, CUDA 13.4 JIT, cold L2 median) vs FlashAttention-3 BF16:

| Shape | Batch ($B$) | Seq Len ($S$) | Heads ($H$) | Cake End-to-End (ms) | FlashAttention-3 BF16 (ms) | Speedup | Architectural Finding |
|---|---:|---:|---:|---:|---:|---:|---|
| `R1` | 1 | 4,800 | 40 | `0.7530` | `0.6484` | **`0.86x`** | 4-kernel stats+quant overhead exceeds WGMMA gain at short $S$ |
| `R2` | 1 | 32,760 | 40 | `28.9556` | `31.6100` | **`1.09x`** | Quadratic $O(S^2)$ compute amortizes dynamic quantization |
| `R3` | 1 | 75,600 | 40 | `154.6865` | `176.3442` | **`1.14x`** | 946.07 vs 829.89 effective TFLOP/s |
| `R4` | 1 | 32,760 | 5 | `3.7545` | `4.0651` | **`1.08x`** | Low-head-count long sequence stays faster |
| `R5` | 1 | 75,600 | 5 | `19.2451` | `22.0230` | **`1.14x`** | Consistent 1.14x speedup at 75.6K |
| `R6` | 2 | 32,760 | 40 | `58.0676` | `63.9349` | **`1.10x`** | Batch=2 32.7K sequence speedup |
| `R7` | 1 | 32,760 | 12 | `8.7903` | `10.2247` | **`1.16x`** | Peak **1.16x** end-to-end speedup |

### 2.2 FlashInfer `#7370` (`e177fd16c9`): `cake_tinygemm2_tcgen05` Narrow-Tile Unrolling

PR `#7370` unrolls the MMA and TMA loader loops of `cake_tinygemm2_tcgen05`'s narrow tile in blocks of rounds and retunes selection windows across Blackwell targets:
- **`sm_100a` (B200)**: Uses the **2-round** unrolled build, cutting `1x128x14336` latency from `7.58 -> 6.85 µs` (**`1.11x`** with bias) and `7.42 -> 6.88 µs` (**`1.08x`** without bias).
- **`sm_103a` (GB300)**: Uses the **4-round** unrolled build that issues full stages without a per-step $K$ boundary test.
- **`sm_107a` (R200)**: Uses the **4-round** unrolled build.
- Expands each architecture's narrow/packed/accumulator catalog from **12 to 20 programs**.

### 2.3 FlashInfer `#7366` (`5b79a1f222`): `cake_fp4_allreduce` Deslop & Shared Bindings

PR `#7366` compacts the NVFP4-transport all-reduce package across B200 (`sm_100a`), GB300 (`sm_103a`), and R200 (`sm_107a`):
1. Drops **5 unreachable fused-norm prefetch programs** gated at $\ge 212$ CTAs.
2. Folds **14 `_pf` twin kernels** into their base kernels via `-D` preprocessor switches.
3. Replaces 53 per-program bindings with **10 shared bindings** parameterized by `-DCAKE_FP4_ALLREDUCE_KERNEL`, `-DCAKE_FP4_ALLREDUCE_THREADS`, and `-DCAKE_FP4_ALLREDUCE_DYNAMIC_SMEM`.
4. Reduces `.cu` files from **116 to 49** (`39` kernels + `10` bindings), programs from **58 to 53**, modules from **75 to 68**, and cuts **81,881 net lines** (`+2,038 / -83,919`).

### 2.4 FlashInfer `#6231` / `#7193` / `#7105` / `#7420` & TileLang `#3523` / `#3558` / `#2887`

- **cuDNN Bounded Prefill Reuse & Sticky FA2 Graph Schedule (`#6231`)**: Keeps `auto` pinned to FA2 in graph mode once FA2 is planned so subsequent `plan()` calls refresh FA2's CPU-generated output/merge schedule, passing 70 B200 tests, 1,700 lifecycle checks, and 1,000 strict-LSE checks.
- **TileLang Ascend PTO SIMT Vector-Local Offset (`#3523`) & Empty Stepped Loop Guard (`#2887`)**: Scales vector slot indices by `element_lanes` (`SIMTVectorLocalScalarOffset`) and guards `LoopUnroller` with `if (extent == 0) return 0;` before dividing `(extent - 1) / step`.

---

## 3. Deterministic Verification Summary

Running `python experiments/flashinfer_cake_fp8_tcgen05_phase100/run.py` verifies all four architectural invariants (`cake_fp8_long_seq_speedup_band_holds`, `tcgen05_narrow_programs_expanded_12_to_20`, `fp4_allreduce_cu_files_reduced_116_to_49`, and `tilelang_empty_stepped_loop_returns_zero`).
