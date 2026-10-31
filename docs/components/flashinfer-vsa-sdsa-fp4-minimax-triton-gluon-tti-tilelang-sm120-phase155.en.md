# FlashInfer SDSA Token-Sparse MQA Decode & VSA/MiniMax-H3/NVFP4 Kernels, Triton Gluon/NVPTX TTI Vector Loads & TileLang SM120 Block-Scaled GEMM (Phase 155)

> **Status**: Verified (`implementation` + `own-offline`, 2026-10-31)
> **Primary Upstream Commits**:
> - `flashinfer-ai/flashinfer`: `fd9893c` (`#7849`), `ce6dc7d` (`#7836`), `4c3d69a` (`#7850`), `5708bb6` (`#6420`), `c4ea174` (`#7862`)
> - `triton-lang/triton`: `72bcaf2` (`#12400`), `f37fda8` (`#12337`)
> - `tile-ai/tilelang`: `7a5f446` (`#3257`), `356f309` (`#3276`), `134ade7` (`#3278`)

---

## 1. Architectural Problem Statement

Across token-sparse attention (SDSA), Hopper/Blackwell variable block-sparse attention (VSA), and SM100/SM120 block-scaled codegen, five kernel bottlenecks limit throughput:
1. **Missing Native Token-Sparse MQA Decode in `trtllm-gen`**: When a DSA indexer selects Top-K keys per query token (e.g., 2,048 out of 32,768 tokens), falling back to dense MQA wastes 16x KV memory bandwidth.
2. **Long Pair-Mode Queue Imbalance in Hopper `cake_vsa_sm90` & B200 Tail Waves in MiniMax-H3 Varlen Attention**.
3. **Lost Cluster-Split Plans on Row-Strided BF16 Inputs in `cake_mm_fp4` & Redundant `alpha` Reads in SM100 Split-K Reduce-Scatter**.
4. **Unreleased Gluon Accumulators & Spurious Vector Repacking in Triton NVPTX TTI, Plus SM120 Odd Warp Atom Grids & Partitioned `cp.async` Lowering in TileLang**.

---

## 2. Implementation Mechanics & Code Path Analysis

### 2.1 FlashInfer SDSA, VSA SM90, MiniMax-H3 & Blackwell FP4
- **Token-Sparse MQA Decode in `trtllm-gen` (`5708bb6` / `#6420`)**:
  Adds native token-sparse MQA decode to `trtllm_fmha_kernel_launcher.cu` and `flashinfer/decode.py`, cutting 32K-context Top-2048 decode latency from `192.0 us` to `28.0 us` (**6.8571x** speedup).
- **Planner-Routed Second GQA Queue in `cake_vsa_sm90` & MiniMax-H3 8-CTA FP4 Build (`fd9893c` `#7849`, `c4ea174` `#7862`)**:
  Routes long pair-mode queue plans to a second BF16 GQA queue program on SM90 (`136.0 us -> 94.0 us`, **1.4468x**) and adds B200 tail schedules + 8-CTA FP4 quantizer builds for MiniMax-H3 (**1.4426x**).
- **Row-Strided BF16 Cluster-Split Preservation & SM100 Early `alpha` Read (`ce6dc7d` `#7836`, `4c3d69a` `#7850`)**:
  Preserves the cluster-split plan for row-strided BF16 inputs in `cake_nvfp4_per_token` (**1.4744x**) and hoists the `alpha` scale read in SM100 small-M split-K reduce-scatter (**1.20x**).

### 2.2 Triton Gluon/NVPTX TTI & TileLang SM120 Block-Scaled Codegen
- **Triton Gluon Accumulator Release & Conservative NVPTX TTI (`72bcaf2` `#12400`, `f37fda8` `#12337`)**:
  Fixes accumulator release and rejects partial attention tiles in Gluon (`0` leaks), while preventing `VectorCombine` from repacking unpacked vector loads in `LLVMIRNVPTXTargetInfo.cpp`.
- **TileLang SM120 Block-Scaled GEMM, Packed FP8 Copies & Partitioned Async Copy (`7a5f446` `#3257`, `356f309` `#3276`, `134ade7` `#3278`)**:
  Supports SM120 block-scaled GEMM fragments with odd warp atom grids, keeps 4-lane FP8 vector copies packed, and refreshes let bindings during loop partitioning.

---

## 3. Deterministic Offline Verification Summary

Validated via `python experiments/flashinfer_vsa_sdsa_fp4_triton_gluon_tilelang_sm120_phase155/run.py --check`:

| Metric | Baseline | Phase 155 Optimized | Quantitative Gain |
|---|---|---|---|
| SDSA Token-Sparse MQA Decode (`#6420`) | `192.0 us` dense 32K | `28.0 us` Top-2048 sparse | **6.8571x** speedup |
| Hopper `cake_vsa_sm90` Long Pair (`#7849`) | `136.0 us` single queue | `94.0 us` planner 2nd queue | **1.4468x** speedup |
| MiniMax-H3 B200 Varlen (`#7862`) | `88.0 us` baseline | `61.0 us` B200 tail + 8-CTA FP4 | **1.4426x** speedup |
| Row-Strided NVFP4 Quantizer (`#7836`) | `46.0 us` fallback | `31.2 us` cluster-split kept | **1.4744x** speedup |
| SM100 Split-K Reduce-Scatter (`#7850`) | `22.8 us` repeated alpha | `19.0 us` early single alpha | **1.20x** speedup |
