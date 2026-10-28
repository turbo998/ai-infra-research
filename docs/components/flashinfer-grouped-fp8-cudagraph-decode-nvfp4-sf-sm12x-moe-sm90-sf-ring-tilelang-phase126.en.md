# Phase 126 Track A: FlashInfer Grouped FP8 128x64 KG1, CUDA-Graph Decode Merge Skip, NVFP4 Scale-Factor Tile Guard, SM12x Symbolic MoE, SM90 2-Slot Scale Ring & TileLang Cache Isolation

> **Phase 126 (`priority: 135`) · Evidence Class**: Upstream source code and commit descriptions (`implementation` / `author-benchmark`) + deterministic offline analytical calibration (`own-offline`). No physical GPU profiling (`own-hardware`) is claimed.

---

## 1. Core Architectural Mechanisms & Source-Level Analysis

### 1.1 FlashInfer Cake Grouped FP8 GEMM `128 x 64` KG1 One-Wave Route (`#7745`, `4e1b0a75`)
- **Upstream Path**: `csrc/cake_grouped_fp8_gemm/cake_grouped_fp8_gemm_488f15389de98fd92896_kernel.cu`, `flashinfer/gemm/cake_grouped_fp8_gemm.py` (`4e1b0a75058abee792d2fa061408bba83abb308c`).
- **Mechanism**: `cake_grouped_fp8_gemm` consumes FP8 E4M3 inputs and writes BF16 outputs grouped by `m_indices`, supporting 15 FP32 per-128 block-scale routes and 14 packed UE8M0 routes. When `M > 32` and the `128 x 64` tile grid fits within a single SM wave:
  $$\left\lceil \frac{M}{128} \right\rceil \times \left\lceil \frac{N}{64} \right\rceil \le N_{\text{SM}}$$
  calls on the `128 x 128` scale-group-1 (KG1) route move to the new `128 x 64` vector-output KG1 program, doubling N-dimension parallelism within a single wave for a **`1.309x` mean speedup** across `M=64..384, N=2048..4096`.

### 1.2 CUDA-Graph Decode `fixed_uniform_q_len` Merge Skip & NVFP4 Scale-Factor Tile Bounds (`#7020`, `c2f8204b`; `#6024`, `ac38c600`)
- **CUDA-Graph Decode Merge Skip (`#7020`, `c2f8204b`)**:
  - **Root Cause**: Under CUDA graphs, `PrefillBinarySearchKVChunkSize` previously reported `split_kv=true` whenever `enable_cuda_graph` was set, forcing tensor-core decode to write FP32 partial outputs and launch the merge kernel even at batch sizes where no request is ever split.
  - **Fix**: Because tensor-core decode builds `qo_indptr` with a constant `uniform_q_len` across graph replays, passing `fixed_uniform_q_len=true` to `BatchPrefillWithKVCachePlan` clears `split_kv` once query tiles exceed `max_batch_size_if_split`, yielding a **`1.312x` mean speedup** in offline decode modeling.
- **NVFP4 KV Prefill Scale-Factor Tile Guard (`#6024`, `ac38c600`)**:
  - **Root Cause**: In `include/flashinfer/attention/prefill.cuh` (`page_produce_kv_sf` and `produce_kv_sf`), `NUM_SF_ITERS` is rounded up. At `head_dim = 64` with a 64-row KV tile, V scales occupy `SF_TOTAL_BYTES = 256 B`, whereas 128 threads issuing 32-bit loads cover `512 B`. Because the V scale loader uses `kFillZero` `cp.async`, out-of-range lanes with a false predicate still wrote `256` zero bytes directly over the first two 128-byte rows of `q_smem`, causing **`6%–17%` relative error** on query rows 0 and 1 of every CTA.
  - **Fix**: Stopping lanes when `SF_TOTAL_BYTES % (THREADS_PER_CTA * 4) != 0 && flat_byte >= SF_TOTAL_BYTES` eliminates the `256 B` `q_smem` overwrite (`0 B` overwritten, `0.0%` relative error).

### 1.3 SM12x Symbolic Static MoE Compilation, SM120 Direct Micro & Frost Weight-First Scheduling (`#6211`, `c7ec1bca`; `#7502`, `55bf742f`)
- **Symbolic Static MoE Kernel (`#6211`, `c7ec1bca`)**: `_get_static_kernel` in `flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_dispatch.py` previously baked exact token count `m` into compile-time tensor shapes and cache keys (`static_m{m}_...`). Compiling once with `cute.sym_int()` cuts JIT compilation across 24 distinct batch sizes from `108.0 s` to **`4.5 s` (`-95.83%`)**, while extending the SM120 NVFP4 direct micro kernel up to 8 tokens with `_balanced_grid` wave balancing achieves a **`1.448x` mean speedup**.
- **Frost Route Masks & Weight-First TMA (`#7502`, `55bf742f`)**: Directly enumerating set bits of the warp ballot in `route_members.py` avoids per-row ballots/shuffles, while 4 new `bf16-grouped-tc-w8-weight-first-v16` tactics on SM120 issue weight TMA prior to serial activation gathering (`1.192x` speedup).

### 1.4 SM90 NVFP4 2-Slot Scale Ring, Cake Norms/Softmax & TileLang Cache Isolation (`#7542`, `061962ac`; `#7701`, `1a8a0ed0`; `#7702`, `ea2d6e96`; `#7671`, `25957924`; `tilelang#3696`, `0bbdcbfd`)
- **SM90 NVFP4 CuTe DSL MoE EP 2-Slot Scale Ring (`#7542`, `061962ac`)**: Adding a decoupled 2-slot scale ring (`nvfp4_sf_stages=2`) unlocks a 4th AB stage for `N64/K256` NVFP4×FP8 MoE EP, reducing upstream measured latency (`author-benchmark`) from `1,232.6 µs` to **`1,189.6 µs` (`-3.68%`)** at `T=512` and from `1,607.2 µs` to **`1,566.6 µs` (`-2.75%`)** at `T=1024`.
- **Rubin `sm_107a` `cake_muse_spark_norm` (`#7701`, `1a8a0ed0`) & Blackwell `cake_gated_act_mxfp8` (`#7702`, `ea2d6e96`)**: Adds 8-CTA persistent `c8_t128_x8_pers_sc_eo_wc` for `M = 1..128` on `sm_107a` (`1.284x` over non-Cake fallback, 78 total configs) and 8 CTAs/SM packed-sigmoid band-path SwiGLU+MXFP8 programs on `sm_100a`/`sm_103a` (`1.176x`).
- **TileLang `without_kernel_cache()` (`tilelang#3696`, `0bbdcbfd`)**: Routes all cache-sensitive tests through a context manager that restores `CacheState` in `finally`, eliminating `4 -> 0` leaked cache-disabled pytest workers.

---

## 2. Deterministic Offline Calibration Summary (`own-offline`)

Experiment script: `experiments/flashinfer_grouped_fp8_cudagraph_decode_nvfp4_sf_sm12x_moe_sm90_sf_ring_tilelang_phase126/run.py`

| Sub-Model / Metric | Baseline (Before) | Phase 126 Optimization (After) | Verified Outcome |
|---|---|---|---|
| Cake Grouped FP8 GEMM (`M=64..384`) | `128 x 128` KG1 tile | **`128 x 64` KG1 one-wave vector-output** | **`1.309x`** mean speedup (15 FP32 routes) |
| CUDA-Graph Tensor-Core Decode | Unconditional `split_kv=True` + Merge | **`fixed_uniform_q_len=True` merge skip** | **`1.312x`** mean speedup |
| NVFP4 FA2 Prefill (`head_dim=64`) | `kFillZero` overwrites `256 B` of `q_smem` (`17%` err) | **`flat_byte >= SF_TOTAL_BYTES` early break** | **`0 B`** overwritten / **`0.0%`** rel. error |
| SM12x `b12x` Static MoE (24 batch sizes) | Per-`m` `static_m{m}` recompile (`108.0 s`) | **`cute.sym_int()` single compile (`4.5 s`)** | **`-95.83%`** JIT compile time; Direct Micro **`1.448x`** |
| SM90 NVFP4 MoE EP (`T=512 / 1024`) | 3 AB stages (`1232.6 / 1607.2 µs`) | **`nvfp4_sf_stages=2` unlocks 4 AB stages** | **`1189.6 µs (-3.68%)`** / **`1566.6 µs (-2.75%)`** |
| TileLang Cache-Sensitive Workers | `4` workers leak `CacheState.disable()` | **`without_kernel_cache()` restoration** | **`0`** leaked workers |
