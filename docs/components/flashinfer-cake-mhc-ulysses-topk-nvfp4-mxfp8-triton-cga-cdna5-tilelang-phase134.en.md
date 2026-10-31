# Deep Dive: FlashInfer Cake Warp Decode / Rubin `sm_107a` Mega-MHC FP8 Group-128 / SM120 MiniMax-H3 Longest-First Drain / Ulysses World-8 2-Unit Direct Scatter / Per-Stream Top-K `row_states` / NVFP4 Hoisted Reciprocal / MXFP8 Early PDL + Triton CGA DAG Slicing / CDNA5 Hardware `V_TANH` + TileLang Single-Round Cross-Warp AllReduce (Phase 134 · Priority 143)

## 1. Problem Scope & Architecture Overview

This report addresses `q-gpu-flashinfer-cake-mhc-ulysses-topk-nvfp4-mxfp8-triton-cga-cdna5-tilelang-phase134` (Priority 143) using 10 zero-overlap upstream commits merged on 2026-10-30 across `flashinfer-ai/flashinfer`, `triton-lang/triton`, and `tile-ai/tilelang`.

---

## 2. Primary Source Implementation Analysis

1. **FlashInfer Cake Warp Decode `sm_100a` Launch Choices (`flashinfer-ai/flashinfer@36ec610`)**: Switches `e128_i1536_k8` at 8 tokens to the pre-wait work-claiming FC2 sibling (`107.28 -> 106.67 us`, `1.0057x`) and caps `h3072_e256_i768_k8` at 3 tokens to a 128-slot pool (`120` CTAs vs `144`, `22.48 -> 22.12 us`, `1.0163x`).
2. **FlashInfer Rubin `sm_107a` `cake_deepgemm_mega_mhc` FP8 Group-128 Raw-Scale (`flashinfer-ai/flashinfer@7832229`)**: Replaces IEEE `div.rn.f32` with range-guarded `rcp.approx.ftz.f32` + 2 FMA Newton-Raphson steps (`38.0 -> 36.1 us`, `1.0526x`).
3. **FlashInfer Cake MiniMax-H3 `sm_120a` NVFP4 Varlen Longest-First Tail Drain (`flashinfer-ai/flashinfer@386c7c7`)**: Drains the last `4 * grid` units longest-first when key-tile count classes `<= 2` (`148.0 -> 132.5 us`, `1.1170x`).
4. **FlashInfer Cake Ulysses World-8 BF16 Direct Scatter 2-Unit Zone (`flashinfer-ai/flashinfer@10cfb36`)**: Stages the first two 16B units per lane into the tail signal zone and copies both in a single round of warps `1..15` (`21.4 -> 18.5 us`, `1.1568x`).
5. **FlashInfer Multi-CTA Radix Top-K Per-Stream `row_states` Cache (`flashinfer-ai/flashinfer@6336f3e`)**: Replaces per-call `torch.zeros(1 MiB)` across 9 operators with `_get_stream_cache_buf`, cutting per-call launch overhead `4.8 -> 1.2 us` (`4.0x`).
6. **FlashInfer NVFP4 Hoisted Global Reciprocal (`flashinfer-ai/flashinfer@1d2cd26`)**: Computes `reciprocal_approximate_ftz(SFScaleVal)` before the E4M3 zero-scale check while preserving the zero-scale contract (`7.50 -> 7.20 us`, `1.0417x`).
7. **FlashInfer Swizzled MXFP8 Quantizer Early PDL Launch (`flashinfer-ai/flashinfer@80ece84`)**: Signals `griddepcontrol_launch_dependents()` immediately after `griddepcontrol_wait()`, cutting GB200 `M=4, N=6144, K=1024` quant->GEMM graph latency `8.559 -> 8.175 us` (`-4.51%`, `1.0470x`).
8. **Triton `AssignCGALayouts` DAG Slicing (`triton-lang/triton@e9d175a`)**: Slices through shared-producer DAGs instead of trees, eliminating `2 -> 0` redundant `ttg.convert_layout` ops when a `tt.descriptor_load` feeds multiple `tt.dot` consumers.
9. **Triton AMD CDNA5 (`gfx1250`) Hardware `V_TANH` (`triton-lang/triton@8abae58`)**: Lowers `fast_tanhf` to `llvm.amdgcn.tanh.f32` (`8 -> 1` instructions, `8.0x`) while keeping `libdevice.tanh` exact via OCML.
10. **TileLang Single-Round Cross-Warp AllReduce (`tile-ai/tilelang@0176ba1`)**: Collapses cross-warp reductions into 1 shared-memory round and 2 barriers regardless of warp count, cutting B300 `per_block_cast` `BAR.SYNC` `12 -> 4` (`-66.7%`) and `STS` `6 -> 2` (`-66.7%`).

---

## 3. Reproducible Offline Verification

- **Command**: `python experiments\flashinfer_cake_mhc_ulysses_topk_nvfp4_mxfp8_triton_cga_cdna5_tilelang_phase134\run.py --check`
- **Artifact**: `experiments/flashinfer_cake_mhc_ulysses_topk_nvfp4_mxfp8_triton_cga_cdna5_tilelang_phase134/results/summary.json`
