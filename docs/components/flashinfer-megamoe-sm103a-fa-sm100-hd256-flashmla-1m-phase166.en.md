# FlashInfer SM103a MegaMoE v3 TP/EP Rank Retuning & Windows `interface` Macro Fix, FlashAttention CuTe SM100 `hd256` 2-CTA Alignment & `seqused` Support, and FlashMLA >1M Grid Reordering (Phase 166)

> **中文版**: [flashinfer-megamoe-sm103a-fa-sm100-hd256-flashmla-1m-phase166.md](flashinfer-megamoe-sm103a-fa-sm100-hd256-flashmla-1m-phase166.md)

## 1. Primary Source Audit Scope (Track 1 · 11 Zero-Overlap Commits)

- **FlashInfer**: `#8133` (`5086eb9db8`, 2026-10-31), `#7950` (`3817e57a7a`, 2026-10-30), `#7878` (`90988f59eb`, 2026-10-30), `#7895` (`f5cb50d624`, 2026-10-29)
- **FlashAttention**: `#2880` (`98eb7998a0`, 2026-09-11), `#2810` (`0f3fb00d3f`, 2026-09-11)
- **CUTLASS**: `9e6adccdbf` (2026-08-05)
- **Triton**: `#12410` (`f7bec9df62`, 2026-10-21), `#12407` (`f7f5641b4a`, 2026-10-21)
- **FlashMLA**: `#182` (`9241ae3ef9`, 2026-04-29), `#173` (`71c737929f`, 2026-03-31)

## 2. Core Architectural Mechanisms & Quantitative Verification

1. **FlashInfer `cake_mega_moe_v3` SM103a TP/EP Rank Retuning & Windows `interface` Macro Fix (`#8133`, `#7895`)**:
   - Retunes `9` of `32` TP/EP rank programs on `sm_103a` for the 384-expert, top-6, hidden-5120 single-launch pipeline (dispatch + FC1 + clamped SwiGLU + MXFP8 + FC2 + weighted combine), reducing latency from `48.00 us` to `41.28 us` (**-14.0%**), and renames the `interface` parameter in `trtllm_gemm_runner.cu` to eliminate Windows `<objbase.h>` macro collisions (**0** build errors).
2. **FlashAttention CuTe SM100 `hd256` 2-CTA 16B Vectorized Store Alignment & `seqused_q`/`seqused_k` Paged-KV Support (`#2880`, `#2810`)**:
   - Preserves 16-byte output alignment in `sm100_hd256_2cta_fmha_forward` (**4.0x** wider store width vs 4B scalar stores) and adds `seqused_q`/`seqused_k` support for vLLM continuous-batching paged-KV decode steps.
3. **FlashMLA `>1M` Sequence Backward Convert Grid Reordering & Combine Grid Swap (`#173`, `#182`)**:
   - Reorders grid dimensions in `fmha_kernel_bwd_convert.hpp` so sequence tile blocks (`65,536` at 4M tokens) map to `gridDim.x` instead of exceeding CUDA's `65,535` `gridDim.y/z` limit (**0** grid overlimit faults).

## 3. Deterministic Reproduction

```bash
python experiments/flashinfer_megamoe_sm103a_fa_sm100_hd256_flashmla_1m_phase166/run.py --check
```
