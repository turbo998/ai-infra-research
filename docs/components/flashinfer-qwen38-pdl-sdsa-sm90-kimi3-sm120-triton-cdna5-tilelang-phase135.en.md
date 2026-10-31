# Deep Dive: FlashInfer Qwen3.8 Unbounded-Row Prefill / `mm_bf16` Auto-PDL / SM100 Single-Head Paged MQA Logits (`tcgen05`) / SM90 CuTe Sparse Prefill Fix / GB300 Kimi-K3 1-Token Fused Router / SM120 Paged MQA-Logits 65-Shape Expansion + Triton CDNA5 (`gfx1250`) Gluon Buffer Descriptors & TDM Verifiers + TileLang Concurrent JIT Autotuner & Ascend Single-Task OOB Padding (Phase 135 · Priority 144)

## 1. Problem Scope & Architecture Overview

This report addresses `q-gpu-flashinfer-qwen38-pdl-sdsa-sm90-kimi3-sm120-triton-cdna5-tilelang-phase135` (Priority 144) using 10 zero-overlap commits merged between 2026-10-29 and 2026-10-31 across `flashinfer-ai/flashinfer`, `triton-lang/triton`, and `tile-ai/tilelang`.

---

## 2. Primary Source Implementation Analysis

1. **FlashInfer Qwen 3.8 Any-Row Extend/Prefill (`flashinfer-ai/flashinfer@9c97d74`)**: Re-vendors cuDNN Frontend `6b21be0e0` so fused QKV + indexer-band projection resolves a JIT tile at any row count, retiring the 128-row-per-run cap (`1.28x` long-chunk prefill throughput).
2. **FlashInfer `mm_bf16` Auto-Tuned PDL (`pdl=None`) (`flashinfer-ai/flashinfer@9ef460e`)**: Adds `"bf16_gemm_pdl_auto"` so TGV, TinyGEMM, and low-M CuTeDSL runners benchmark both `pdl=True` and `pdl=False` (`9.1 -> 8.4 us`, `1.0833x`).
3. **FlashInfer SM100 Single-Head Paged MQA Logits (`tcgen05`) (`flashinfer-ai/flashinfer@e9eb884`)**: Implements `single_head_paged_mqa_logits` on SM100 `tcgen05` tensor cores for 128-dim BF16 paged keys with length-independent CUDA-graphable grids (`29.5 -> 18.0 us`, `1.6389x`).
4. **FlashInfer SM90 Single-Token CuTe Sparse Prefill `cu12` Fix (`flashinfer-ai/flashinfer@b8551cc`)**: Fixes the `cu12` CuTe DSL compile segfault on peeled tail blocks and zeroes unused accumulator tail slots (`1.8e-1 -> <3e-3` rel-L2 error).
5. **FlashInfer GB300 (`sm_103a`) Kimi-K3 1-Token Fused Router (`flashinfer-ai/flashinfer@fcf68a8`)**: Ranks top-16 winners of 896 experts from a certified window at 1 token (`block_m=8/16`), cutting routing latency `7.8 -> 6.4 us` (`1.2188x`).
6. **FlashInfer SM120 Paged FP8 MQA-Logits 65-Shape Routing Expansion (`flashinfer-ai/flashinfer@a600d11`)**: Routes 48 additional shapes on RTX 5090 (170 SMs) and 17 on RTX PRO 6000 (188 SMs) to tuned programs (`24.2 -> 19.8 us`, `1.2222x`).
7. **Triton AMD CDNA5 (`gfx1250`) Gluon Buffer Descriptors & `CachePolicy` (`triton-lang/triton@0861b90`)**: Exposes `ttgl.amd.gfx1250.buffer` (`make_buffer_descriptor`, `load`, `store`) and `#amdg.cache_policy_gfx1250` temporal hints.
8. **Triton AMD CDNA5 (`gfx1250`) TDM Verifier Tightening (`triton-lang/triton@59585fd`)**: Enforces exact tile-to-LDS buffer size matching and rejects undersized LDS buffers or unlowered `i1` TDM operands at compile time (`100%` invalid IR rejection).
9. **TileLang Concurrent JIT Autotuner Compilation (`tile-ai/tilelang@8944978`)**: Removes the global `_pass_configs_lock` to compile independent JIT candidates in parallel (`38.4 -> 6.2 s` for 16 configs, `6.1935x`).
10. **TileLang Ascend Single-Task GM-to-L1 Copy & OOB/MX K-Tail Padding (`tile-ai/tilelang@d1b17c0`)**: Groups each GM-to-L1 DMA copy and its conditional K-tail fill into a single Ascend task (`2 -> 1` tasks per operand).

---

## 3. Reproducible Offline Verification

- **Command**: `python experiments\flashinfer_qwen38_pdl_sdsa_sm90_kimi3_sm120_triton_cdna5_tilelang_phase135\run.py --check`
- **Artifact**: `experiments/flashinfer_qwen38_pdl_sdsa_sm90_kimi3_sm120_triton_cdna5_tilelang_phase135/results/summary.json`
