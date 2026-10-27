# Phase 112 Track A: FlashInfer Rubin SM107 MegaMoE Masked-Route NaN Staging, SM120 MXFP8×MXFP4 TMA 128B L2 Promotion, Cake 256×448 MXFP8 FC1 & CuTe IR / TileLang Pipeline (2026-10-28)

> **中文对照 / Chinese Version**: [`flashinfer-rubin-megamoe-mxfp8-l2-cutlass-cuteir-tilelang-phase112.md`](./flashinfer-rubin-megamoe-mxfp8-l2-cutlass-cuteir-tilelang-phase112.md)  
> **7-Card 4:5 Visual Benchmark Report**: [`../reports/ai-infra-phase112-benchmark-cards.html`](../reports/ai-infra-phase112-benchmark-cards.html)  
> **Reproducible Experiment Suite**: [`../../experiments/flashinfer_rubin_megamoe_mxfp8_l2_cutlass_cuteir_tilelang_phase112/`](../../experiments/flashinfer_rubin_megamoe_mxfp8_l2_cutlass_cuteir_tilelang_phase112/)

---

## 1. Evidence Classification & First-Hand Commit Ledger (10 Kernel & Compiler Commits)

| # | Repo & Commit | Date | Evidence Tier | File Paths & Core Mechanism |
|---|---|---|---|---|
| 1 | [`flashinfer@7bce78b15d`](https://github.com/flashinfer-ai/flashinfer/commit/7bce78b15d6462e054bbd565fbadaf4daa65ac0d) | 2026-10-27 | `implementation` | `flashinfer/moe_ep/backends/mega/kernel/sm107/{routing_staging,routing_validation}.py`: Rubin (`sm_107a`) MegaMoE tolerates `NaN`/`±inf` scores on masked (`expert == -1`) routes and zeroes them into the workspace via `RouteStager` without mutating caller tensors; adds `FLASHINFER_SM107_MEGA_CHECK_ROUTES=0` |
| 2 | [`flashinfer@a1fbb4e0c4`](https://github.com/flashinfer-ai/flashinfer/commit/a1fbb4e0c4703c65a108bff1a17be5067a72adf0) | 2026-10-26 | `author-benchmark` / `implementation` | `flashinfer/fused_moe/backends/cudnn_frost/artifacts/mxfp8_mxfp4/sources/*.py`: Enables CUTLASS default `TensorMapL2Promotion.l2_128b` on SM120 cuDNN FROST `MXFP8 × MXFP4` TMA descriptors for 64B packed E2M1 K-rows (`0.85 → 1.19 TB/s`, `343 → 253 µs` on RTX PRO 6000) |
| 3 | [`flashinfer@fd0e4fdd96`](https://github.com/flashinfer-ai/flashinfer/commit/fd0e4fdd96b046b1433a44e9eec8f0c3d8fe9e70) | 2026-10-25 | `author-benchmark` / `implementation` | `csrc/cake_minimax_h3_mlp_sm10x.cu`, `flashinfer/diffusion_ops/minimax_h3_mlp.py`: Adds `256 × 448` MXFP8 FC1 program with staggered accumulator halves on SM100/SM103, cutting N-tiles from `112` to `64` (`1.023x–1.060x` on B200, `1.033x–1.068x` on B300) |
| 4 | [`flashinfer@cc8552a3a0`](https://github.com/flashinfer-ai/flashinfer/commit/cc8552a3a0158391f514571597b3b495090ce43c) | 2026-10-25 | `implementation` | `flashinfer/experimental/cake_qsa_nvfp4/{cake_backend,cake_jit}.py`: Adds Rubin SM107 (`sm_107a`) `tcgen05` decode and merge programs for `cake_qsa_nvfp4`, retaining the split (`decode + merge`) launch form on all hosts |
| 5 | [`flashinfer@4796d2effd`](https://github.com/flashinfer-ai/flashinfer/commit/4796d2effd9ea8511c32ffcd21609998c2896bc0) | 2026-10-25 | `author-benchmark` / `implementation` | `csrc/cake_minimax_h3_sm120_quant_out_proj_sm120a.cu`: Computes `alpha = 1 / (act_global_scale * o_weight_global_scale)` in FP64 on the device for SM120 NVFP4 out-projection, removing 2 host syncs per call and shrinking back-to-back GPU idle gaps from `186.9–223.3 µs` to `3.1–3.6 µs` |
| 6 | [`flashinfer@6185549792`](https://github.com/flashinfer-ai/flashinfer/commit/618554979285eaf2fa803e6803e1ccb014efd957) | 2026-10-27 | `implementation` | `flashinfer/{decode,prefill,sparse}.py`: Calls `.contiguous()` on `indptr`, `indices`, `last_page_len`, and `seq_lens` in `plan()`, `fast_decode_plan()`, and `workspace_size()` to prevent strided-view CUDA/CPU layout divergence and illegal memory accesses |
| 7 | [`cutlass@9d79de097b`](https://github.com/NVIDIA/cutlass/commit/9d79de097be048b67be8e527ceced4ba017e5e1d) | 2026-08-03 | `implementation` | `cutlass_compiler/README.md`: Introduces the `CuTe IR` MLIR dialect stack (`cute` dialect + `Base` pseudo-dialect boundary + `cute-to-base` lowering pass) |
| 8 | [`cutlass@f94ec46f4f`](https://github.com/NVIDIA/cutlass/commit/f94ec46f4f63f96003d6cfdf2014731e7672c281) | 2026-07-29 | `implementation` | `include/cutlass/gemm/kernel/sm100_tile_scheduler_group.hpp`: Removes dead `load_query_response` wrapper from `PersistentTileSchedulerSm100Group` |
| 9 | [`tilelang@fc0fbf0e50`](https://github.com/tile-ai/tilelang/commit/fc0fbf0e50de55c8e3f04b4954c2ab1000b8a298) | 2026-10-16 | `implementation` | `src/transform/inject_pipeline.cc`: Prevents double 2D→3D→4D expansion of shared buffers across sibling `T.Pipelined` loops with asymmetric `num_stages` via `PipelineBufferReconciler` |
| 10 | [`tilelang@51bdf5467e`](https://github.com/tile-ai/tilelang/commit/51bdf5467e560de35fdc91e88821549480654180) | 2026-10-16 | `author-benchmark` / `implementation` | `src/ascend/transform/ascend_simdvf_lower_parallel.cc`: Fixes Ascend `SimdVF` `dim=1` reduction store to emit a 1-element `ONEPT_B32` (`vsts_1st`) at `dst[rep]` instead of a 64-lane `NORM_B32` store at `dst[rep * vreg_size_]` |

---

## 2. Architectural Mechanisms & Formal Formulations

### 2.1 Rubin SM107 MegaMoE Masked-Route Non-Finite Score Staging (`flashinfer@7bce78b15d`)

When upstream routers pad unused slots with `expert == -1`, the corresponding score slot may hold `NaN` or `±inf`. The updated validator relaxes finiteness strictly on masked slots:

$$\text{valid}_{\text{fixed}} = (-1 \le e < E) \land \big((e = -1) \lor \text{isfinite}(s)\big) \land \neg\text{duplicate}(e)$$

Simultaneously, the CuTe-DSL `RouteStager` copies prequantized NVFP4/MXFP8 routing tensors into the MegaMoE workspace while substituting `0.0f` whenever $e = -1$, preserving caller tensors untouched and preventing `NaN` propagation during late-weight combine.

### 2.2 SM120 cuDNN FROST `MXFP8 × MXFP4` TMA 128B L2 Promotion (`flashinfer@a1fbb4e0c4`)

In `MXFP8 × MXFP4` (W4A8) grouped MoE GEMM on SM120 (`sm_120a`), each 128-element FP8 K-tile corresponds to a 64-byte packed E2M1 weight box row ($128 \times 0.5\text{ B} = 64\text{ B}$). Without L2 promotion, every 64B row triggers a separate DRAM request, capping weight stream bandwidth at $0.85\text{ TB/s}$ ($343\text{ }\mu\text{s}$ for 12 experts, `7168×3072`, 16 tokens on RTX PRO 6000). Setting `l2_promotion=_tma.TensorMapL2Promotion.l2_128b` promotes adjacent 64B accesses to 128B L2 sectors, lifting bandwidth to $1.19\text{ TB/s}$ ($+40.0\%$) and cutting kernel time to $253\text{ }\mu\text{s}$ (**$1.3557\times$ speedup**).

### 2.3 Cake MiniMax-H3 `256×448` MXFP8 FC1 & SM120 Device-Scale Out-Projection (`fd0e4fdd96`, `4796d2effd`)

1. **`256×448` Staggered Accumulator Halves (`fd0e4fdd96`)**: Widening the 2-CTA MXFP8 FC1 pair tile from `256×256` ($128$ output columns per pair, $112$ N-tiles) to `256×448` ($224$ output columns per pair, $64$ N-tiles—a **$42.86\%$ reduction**) with staggered TMEM accumulator halves yields a $1.023\times\text{–}1.060\times$ speedup on B200 ($0.9300 \to 0.8773\text{ ms}$ at $M=4816$) and $1.033\times\text{–}1.068\times$ on B300.
2. **Device-Side FP64 Scale in SM120 NVFP4 Out-Projection (`4796d2effd`)**: Computing $\alpha = 1 / (s_{\text{act}} \cdot s_W)$ in FP64 on the GPU eliminates two `.item()` host synchronizations per call, shrinking the inter-call GPU idle gap from $186.9\text{–}223.3\text{ }\mu\text{s}$ to $3.1\text{–}3.6\text{ }\mu\text{s}$ (**$61.22\times$ reduction** at midpoint).

---

## 3. Deterministic Offline Verification Results (`own-offline`)

Validated via `python experiments/flashinfer_rubin_megamoe_mxfp8_l2_cutlass_cuteir_tilelang_phase112/run.py --check` ([`summary.json`](../../experiments/flashinfer_rubin_megamoe_mxfp8_l2_cutlass_cuteir_tilelang_phase112/summary.json)):

| Dimension | Legacy / Baseline | Fixed / Optimized | Key Quantitative Metric |
|---|---|---|---|
| Rubin MegaMoE Masked Route Check | `legacy_check_passed = False` | `fixed_check_passed = True` | Masked scores staged as `0.0`; caller tensor unmodified |
| SM120 `MXFP8 × MXFP4` TMA L2 Promotion | `0.85 TB/s`, `343.0 µs` (`none`) | `1.19 TB/s`, `253.0 µs` (`l2_128b`) | **`1.40x`** bandwidth gain, **`1.3557x`** speedup |
| Cake MiniMax-H3 `256×448` MXFP8 FC1 | `112` N-tiles (`0.9300 ms` @ M=4816) | `64` N-tiles (`0.8773 ms` @ M=4816) | **`42.86%`** fewer N-tiles, **`1.0601x`** speedup |
| SM120 NVFP4 Out-Proj Device Scale | 2 host syncs/call, `205.1 µs` gap | Device FP64 $\alpha$, `3.35 µs` gap | **`61.22x`** smaller inter-call GPU idle gap |
| TileLang Sibling Pipeline & Ascend Reduce | 4D crash / OOB offsets `[64, 128, 192]` | 3D `(4, 64, 32)` / `ONEPT_B32` `dst[rep]` | **`0.0`** max abs error on Ascend `A(4,64)` |
