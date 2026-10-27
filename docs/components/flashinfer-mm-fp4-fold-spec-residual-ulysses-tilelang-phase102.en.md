# FlashInfer `cake_mm_fp4` Cluster-Split & Token-Remainder Folds, `ChainSpeculativeSampling` Deferred Residual Reduction, Cake Ulysses World-2 Direct All-to-All & TileLang TMA Stream Binding (Phase 102)

- **Question ID**: `flashinfer-mm-fp4-fold-spec-residual-ulysses-tilelang-phase102` (`priority: 111`, component: `gpu-software-stack`)
- **Evidence Status**: `own-offline` (First-hand upstream commit inspection + deterministic Python 3.12 stdlib model `experiments/flashinfer_mm_fp4_fold_spec_res_ulysses_phase102/run.py`)
- **Chinese Companion**: [`flashinfer-mm-fp4-fold-spec-residual-ulysses-tilelang-phase102.md`](flashinfer-mm-fp4-fold-spec-residual-ulysses-tilelang-phase102.md)

## 1. Primary Upstream Sources (Zero-Overlap)

1. **FlashInfer `cb138f8aa6` (#7550)**: `perf(cake_mm_fp4): cluster-split per-token quantizer and token-remainder GEMM folds on B200 and GB300`
2. **FlashInfer `03bd5b3402` (#5201)**: `perf(sampling): defer chain speculative residual reduction`
3. **FlashInfer `88205170de` (#7566)**: `perf(cake_ulysses): faster world-2 direct all-to-all, symmetric handshake`
4. **FlashInfer `62b5ae4e52` (#7336)**: `fix(ffi): represent absent functions and MoE tensors explicitly`
5. **FlashInfer `8e7a21d538` (#7565)**: `test(cake_gdn): collect garbage before the GDN prefill allocation-count window`
6. **NVIDIA CUTLASS `c8fc8a6535` (#3666)**: `[CuTeDSL] docs: fix cutlass.range unroll=0 and unroll_full descriptions`
7. **NVIDIA CUTLASS `ee98dbdfd5` (#3693)**: `Keep plain pytest functions in separate sharding groups`
8. **NVIDIA CUTLASS `218d3bfc4a` (#3720)**: `[CuTeDSL] docs: fix local_tile example output shapes`
9. **TileLang `837486abf7` (#3365)**: `Fix constant row indices for TMA gather and scatter`
10. **TileLang `0d75cd9af5` (#3359)**: `[BugFix][CUDA] Honor tensor-device streams in NVRTC and CuTeDSL`

---

## 2. Core Implementation Mechanics & Quantitative Analysis

### 2.1 FlashInfer `cake_mm_fp4` Cluster-Split Quantizer & Token-Remainder GEMM Folds (`cb138f8aa6`)

On Blackwell (`sm_100a` B200 and `sm_103a` GB300), `cake_mm_fp4` performs per-token NVFP4 activation quantization followed by an NVFP4×NVFP4 GEMM:
- **Standalone Quantizer 2/4/8-CTA Cluster Split (`QUANT_ALONE_CLUSTER_SPLIT`)**: When `nvfp4_quantize(..., per_token_activation=True, backend="cake")` runs standalone on wide rows (`K ∈ {16384, 18432, 28672}`) at small token counts (`M=1..8` or `M=130`), assigning a single CTA per row under-utilizes SMs. Splitting each row across clusters of `C ∈ {2, 4, 8}` CTAs yields **1.027x–1.085x** speedups across 14 B200 cells (geomean **1.0579x**) and **1.050x–1.086x** across 12 GB300 cells (geomean **1.0649x**).
- **Token-Remainder GEMM Folds (`K×N = 7168×18432`)**:
  - For `M ∈ [129, 136]`, `main` launched a half-empty 2-CTA 256-token pair tile; folding the `1..8` remainder rows into a single **1-CTA 128-token tile** delivers **1.028x–1.059x** speedups.
  - For `M ∈ [257, 272]`, `main` launched two 2-CTA 256-token pair tiles where the second wave carried only `1..16` rows; folding those remainder rows into a single **2-CTA 256-token pair tile** achieves **1.233x** GEMM speedup on B200 (`25.91 → 21.00 µs`) and **1.194x** on GB300 (`23.82 → 19.94 µs`).

### 2.2 `ChainSpeculativeSampling` Deferred Residual Reduction (`03bd5b3402`)

`ChainSpeculativeSampling` computes the normalizing sum of $\text{relu}(q - p)$ across vocabulary tiles of size $1024 \times 4 = 4096$. Because nothing inside the loop reads the running total, accumulating in thread-local registers (`threadlocal_sum_relu_q_minus_p`) and running `BlockReduce` once after the loop replaces **32** block reductions and barriers with **1** (`32.0x` reduction for $V=128{,}256$), delivering **1.13x–1.66x** speedup on A100 (22.5% mean latency reduction) and **1.05x–1.65x** on H100 (20.1% mean reduction).

### 2.3 Cake Ulysses World-2 Direct All-to-All & TileLang TMA/Stream Fixes (`88205170de`, `837486abf7`, `0d75cd9af5`)

- **Cake Ulysses Direct Route (`88205170de`)**: Streams peer blocks across **128 CTAs** while **20 CTAs** concurrently copy the local block with an unconditional symmetric `ld.acquire.sys.global.u32` handshake, reaching **1.039x** on B200 (`134.04 → 128.98 µs`) and **1.030x** on B300 (`134.45 → 130.50 µs`).
- **TileLang TMA & Stream Fixes (`837486abf7`, `0d75cd9af5`)**: Normalizes Python `int` row indices into `tirx.const(row, dtype="int32")` for `tma_gather4`/`tma_scatter4` and queries `torch.cuda.current_stream(first_tensor.device)` in NVRTC/CuTeDSL adapters.
