# Phase 111 Track A: FlashInfer Frost W8A16 MoE & Hopper MSA Top-K, FA4 SM100 Sparse MLA Recompute-P & Token-Chunked Backward, and TileLang `call_extern` Bare Data Var Write Marking

> **Provenance & Evidence Boundary**: Built from first-party public GitHub commits and our deterministic stdlib simulation (`experiments/flashinfer_frost_w8a16_fa4_sparse_mla_tilelang_phase111/run.py`), strictly separating **Specification**, **Implementation**, **Author Benchmark**, and **Own-Offline** verification without claiming live GPU profiling on SM90/SM100/SM103/SM120 hardware.

---

## 1. FlashInfer Kernel Evolution: Frost W8A16 MoE, Hopper MSA Top-K, Dynamic EP Dispatch Bound, SM120 NVFP4 Out-Proj, GB300 DSA Shift, SM12x >2GiB Static DLPack & Non-64 SSU

### 1.1 Frost Low-Latency Grouped W8A16 MoE Candidates (`flashinfer-ai/flashinfer@d42c276bd872`, 2026-10-27)
- **Implementation Path**: `flashinfer/jit/gemm/moe/gen_moe_w8a16_module.py` expands grouped `E4M3` weight + `BF16/FP16` activation Tensor Core candidates across three GPU generations:
  - **SM90 (`sm_90a`)**: Adds `K256` tiles for `T4/T8` BF16 GeGLU shapes;
  - **SM103 (`sm_103a`, GB300)**: Enables `serial_gather=True` at `T4` BF16 GeGLU and covers `T8/H7168/I2048/K8` topologies;
  - **SM120 (`sm_120a`, RTX 5090 / RTX PRO 6000)**: Adds `group8 / stages2 / K128` candidates.

### 1.2 Hopper `cake_hopper_msa` Single-Pass Exact Top-K Block Select (`flashinfer-ai/flashinfer@3d01e3fec63f`, 2026-10-27)
- **Implementation Path**: Regenerates three `msa_topk_select` candidates onto the single-pass exact schedule (`276e272a54b4efd20df6` plain 1-chunk 128-wide 11-level, `7bbffa90057de145b905` masked 1-chunk 128-wide 11-level, and `e9b601567cc3a6dd8033` masked 2-chunk 32-wide 9-level latency form), achieving up to `1.06x` author-reported speedup on H100.

### 1.3 `MoEEpCommunication.dispatch` Dynamic Per-Step Token Bound (`flashinfer-ai/flashinfer@47e64b0e96c4`, 2026-10-27)
- **Implementation Path**: Threads `runtime_max_tokens_per_rank: Optional[int] = None` through `MoEEpCommunication.dispatch()` and `MoEEpSplitLayer`, enforcing $\text{num\_tokens} \le \text{runtime\_max\_tokens\_per\_rank} \le \text{max\_tokens\_per\_rank}$. When runtime decode batch (`512`) is smaller than the allocated pool ceiling (`4096`), per-step dispatch span drops by `8.0x` without buffer reallocation.

### 1.4 SM120 MiniMax-H3 `quant_out_proj` NVFP4 Routing & GB300 DSA Indexer Threshold Shift
- **SM120 NVFP4 170-SM Routing (`flashinfer-ai/flashinfer@73c83aac3125`, 2026-10-27)**: Extends `PROGRAM_1_SMS = {0: (188,), 1: (188, 170)}` in `cake_minimax_h3_sm120_quant_out_proj.py` so 170-SM RTX 5090 GPUs run `Program 1` (`warp_epilogue`) for NVFP4 (`quant=1`), reading each activation-scale row once per pair of ring stages (`346.3 -> 338.8 us` at `M=4097`, `2316.9 -> 2267.1 us` at `M=33472`).
- **GB300 `cake_dsa_indexer` Shift Rule (`flashinfer-ai/flashinfer@2b5a5ffff99a`, 2026-10-27)**: Sets `sample_shift_rule = [4096, 375]` on `sm_103a` (GB300), shifting the sampled initial threshold by `375‰` instead of `250‰` when `top_k >= 4096` (`1.13x` author-reported speedup on `Z1k4096`).

### 1.5 SM12x W4A16 MoE `> 2 GiB` Static CuTe DLPack & Non-Multiple-of-64 `head_dim` SSU
- **SM12x W4A16 `> 2 GiB` Weight Bank (`flashinfer-ai/flashinfer@9c4aacf1bfd6`, 2026-10-26)**: Wraps flat `uint8` weight banks (`2,684,354,560 B` for 256-expert Qwen3.6-35B-A3B-NVFP4, exceeding `2^31 - 1`) in static-layout `cute.runtime.from_dlpack(w13_arg, assumed_align=16)` to avoid 32-bit signed stride `OverflowError`.
- **Non-Multiple-of-64 `head_dim` SSU (`flashinfer-ai/flashinfer@0c3d03c7656a`, 2026-10-27)**: Enforces `kDim % CTAS == 0 && (kDim / CTAS) % kMinRows == 0` (`kMinRows = 16`) in `dispatchCtasPerHead` (`head_dim=96 -> 2` CTAs, `head_dim=80 -> 1` CTA) and guards vectorized epilogues with `elemsPerThreadEpilogue % load_output_t::count == 0`.

---

## 2. FlashAttention-4 SM100 Sparse MLA In-Kernel Recompute-P & Token-Chunked Backward (`294cfb3593ed`, `#2816`) and `HD256` Varlen Scheduling (`38ddafeb35e5`, `#2807`)

- **`gather_bwd_recompute_p=True`**: Eliminates saving unnormalized `p` (`512 KiB/token` = **32.0 GiB** at `64K` tokens) and `row_max` (`8 KiB/token`) in forward (`8K` forward drops from `4.43 ms` to `3.77 ms`), reconstructing $P^T = \exp_2(\text{scale}\cdot\log_2 e \cdot S^T - \text{lse}\cdot\log_2 e)$ in-kernel via an extra $S^T$ UMMA while zero-filling `sV` gather stages at kernel launch to prevent `0 * NaN` hazards on masked top-k slots.
- **`gather_bwd_token_chunk=N`**: Runs the three backward kernels in token chunks (`N=4096` bounds the `dS` transient to **2.0 GiB** instead of **32.0 GiB** at `64K` tokens) and shrinks each causal chunk's K limit to $k_{\text{end}} = \text{seqlen\_k} - \text{seqlen\_q} + \text{tok}_1$.
- **SM100 `HD256` Varlen Grid Bound (`38ddafeb35e5`)**: Bounds `head_dim=256` 2-CTA FMHA grids by `max_seqlen_q` or dispatches `SingleTileVarlenScheduler` when host maxima are absent, eliminating `batch_size * ceil(total_q / tile_m)` CTA over-launches.

---

## 3. TileLang `AnnotateReadOnlyParams` Fix for Bare Data Vars in `call_extern` (`tile-ai/tilelang@f3ffeaa6b8ee`, `#2618`)

- Marks bare parameter/buffer `.data` `VarNode` arguments passed directly to `builtin::call_extern()` as potentially written in `src/transform/annotate_read_only_params.cc`, preventing invalid `const` pointer emission in CUDA codegen while preserving read-only optimization for pure intrinsics (`tirx.isnullptr`).
