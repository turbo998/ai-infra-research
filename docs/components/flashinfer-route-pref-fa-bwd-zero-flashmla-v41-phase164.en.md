# FlashInfer MoE Finalize+AllReduce Route Prefetch & Deterministic Naming, FlashAttention CuTe Bwd Empty Q/K & FlashMLA DeepSeek-V4.1 256-Splits (Phase 164 · Priority 173)

> **中文版本**: [`flashinfer-route-pref-fa-bwd-zero-flashmla-v41-phase164.md`](flashinfer-route-pref-fa-bwd-zero-flashmla-v41-phase164.md)

## 1. Primary Source Evidence & Classification

| Source ID | Repository & Commit | Release Date | Evidence Tier | Key Code Path & Architectural Mechanism |
| :--- | :--- | :--- | :--- | :--- |
| `src-flashinfer-moe-finalize-allreduce-route-prefetch-2026` | `flashinfer-ai/flashinfer` `#8073` (`795601c070`) | 2026-10-31 | `implementation` | `csrc/experimental/moe_finalize_allreduce_fusion/*`: Issues the first token's route indices ahead of control words, cutting control-word stall cycles by 75% |
| `src-flashinfer-revert-sm120-nvfp4-longest-first-2026` | `flashinfer-ai/flashinfer` `#8075` (`5f1eae4ddc`) | 2026-10-31 | `implementation` | `flashinfer/experimental/cake_minimax_h3/*`: Reverts longest-first drain order for two-class overlapped SM120 NVFP4 varlen attention plans (`#8040`) |
| `src-flashinfer-dsa-indexer-det-names-2026` | `flashinfer-ai/flashinfer` `#8061` (`7078416325`) | 2026-10-31 | `implementation` | `csrc/experimental/dsa_indexer_det/*`: Enforces deterministic symbol/package names for the DSA indexer top-k package |
| `src-flashinfer-rmsnorm-silu-autograd-det-names-2026` | `flashinfer-ai/flashinfer` `#8045` (`c677b26a0f`) | 2026-10-31 | `implementation` | `csrc/experimental/rmsnorm/*`: Enforces deterministic package names for RMSNorm and SiLU-gated RMSNorm autograd packages |
| `src-flashattn-cute-bwd-empty-qk-zero-grad-2026` | `Dao-AILab/flash-attention` `#2776` (`008e63667e`) | 2026-09-14 | `implementation` | `flash_attn/cute/flash_bwd.py`: Returns explicit zero gradients on empty Q/K backward workloads instead of uninitialized memory |
| `src-cutlass-fmha-regression-quick-fix-2026` | `NVIDIA/cutlass` `#3387` (`d01487db82`) | 2026-07-16 | `implementation` | `examples/41_fused_multi_head_attention/*`: Fixes FMHA kernel regression |
| `src-triton-cd-free-threaded-314t-315t-wheels-2026` | `triton-lang/triton` `#12502` (`8fa612157a`) | 2026-10-26 | `implementation` | `.github/workflows/*`: Builds free-threaded Python `3.14t` and `3.15t` wheels |
| `src-triton-amd-new-codegen-build-test-2026` | `triton-lang/triton` `#12391` (`b10226036e`) | 2026-10-21 | `implementation` | `third_party/amd/*`: Adopts new AMD CodeGen LLVM build target and updates regression tests |
| `src-flashmla-st-async-longlong2-msvc-llp64-2026` | `deepseek-ai/FlashMLA` `#224` (`ba89a3466e`) | 2026-09-15 | `implementation` | `csrc/*`: Uses `longlong2` for 128-bit `st.async` PTX transport to prevent 64-bit truncation under MSVC LLP64 |
| `src-flashmla-deepseek-v41-kernels-2026` | `deepseek-ai/FlashMLA` `#221` (`07a1089857`) | 2026-09-10 | `implementation` | `csrc/sm90/*`: Adds specialized MLA kernels for DeepSeek-V4.1 |
| `src-flashmla-decode-combine-num-splits-256-2026` | `deepseek-ai/FlashMLA` `#199` (`15f13e5030`) | 2026-07-28 | `implementation` | `csrc/*`: Extends decode-combine `num_splits` buckets from 128 up to 256 (**`2.0x`**) |

## 2. Deterministic Verification (`experiments/flashinfer_route_pref_fa_bwd_zero_flashmla_v41_phase164`)

- **MoE Finalize+AllReduce Route Prefetch**: Cuts control-word stall cycles by **`75.0%`** (`128 -> 32`) and reduces fused kernel latency from **`19.60 us`** to **`16.46 us`** (**`-16.0%`**).
- **CuTe Bwd Empty Q/K Zero-Grad & FlashMLA 256-Splits**: Eliminates **256** uninitialized NaN elements on empty Q/K backward workloads, restores full **128-bit** `st.async` width on MSVC LLP64 (`64 -> 128 bits`), and expands decode-combine `num_splits` buckets **`2.0x`** (`128 -> 256`).
