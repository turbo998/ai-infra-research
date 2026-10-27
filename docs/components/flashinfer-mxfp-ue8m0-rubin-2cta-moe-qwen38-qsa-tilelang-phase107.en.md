# FlashInfer MXFP4/MXFP8 UE8M0 Subnormal Quantization, Rubin 2-CTA MoE Tactics, Qwen 3.8 QSA Decode & TileLang Cross-Scope Barrier Analysis (Phase 107 · Priority 116)

> **中文版**: [flashinfer-mxfp-ue8m0-rubin-2cta-moe-qwen38-qsa-tilelang-phase107.md](flashinfer-mxfp-ue8m0-rubin-2cta-moe-qwen38-qsa-tilelang-phase107.md)  
> **Evidence Boundary**: `implementation` (first-hand upstream commits) + `own-offline` (deterministic stdlib modeling: `experiments/flashinfer_mxfp_ue8m0_rubin_2cta_moe_qwen38_qsa_phase107/run.py --check`)

## 1. Primary Source Evidence & Implementation Paths

1. **`flashinfer-ai/flashinfer@fb02275` (`#7501`: `csrc/nv_internal/tensorrt_llm/kernels/quantization_utils.cuh`, `flashinfer/quantization/kernels/cutile/mxfp8_grouped_quantize_cutile.py`)**:
   - **Unified MXFP4/MXFP8 Quantization & Bit-Exact Subnormal Handling**: Introduces inline PTX `asm("mul.rn.f32 %0, %1, %2;")` (`mx_quantize_mul`) and `ue8m0_inverse_scale(uint8_t exp)` (`exp == 0 ? 0.0f : __uint_as_float(uint32_t(max(254 - int(exp), 0)) << 23)`) to preserve FP32 subnormals (up to $2^{-127}$) even when compiled under `-use_fast_math`, ensuring a zero UE8M0 scale byte (`e8m0_byte = 0`) deterministically yields signed-zero payloads (`subnormal_inverse_scale = 0.0`) across CUDA, CuTe DSL, and CuTile backends.
2. **`flashinfer-ai/flashinfer@03d7f4a` (`#7196`: `flashinfer/fused_moe/cute_dsl/blackwell/moe_finalize.py`, `moe_w4a16_prepare.py`)**:
   - **Split CuTe DSL W4A16 MoE & Bit-Exact Combine**: Replaces `fma.rn.ftz.f32` with non-FTZ `fma.rn.f32` in `_fma_rn`, separates the initial `_mul_rn` to preserve signed zero, and introduces `moe_permute_w4a16` for bit-exact parity with native MegaMoE combine.
3. **`flashinfer-ai/flashinfer@b0417fd` (`#7027`: `flashinfer/fused_moe/cute_dsl/moe_utils.py`, `tuner.py`)**:
   - **Rubin (SM107) 2-CTA `tile_size=256` & `cluster_shape_n=2` MoE Tactics**: Adds `rubin_moe_cluster_covers_one_tile`. With `tile_size = 256`, `mma_inst_m = 256` (`atom_cta_m = 2`), and `cluster_shape_mn = (2, 2)`, both CTAs along M resolve to the exact same `moe_sort` routing tile (`metadata_padding_tiles_required = 0`) while Cluster N multicasts `A` and `SFA` across CTAs for a **`2.0x` A-operand HBM read reduction**.
4. **`flashinfer-ai/flashinfer@77d35fd` (`#7136`: `docs/experimental/qwen_3_8_block.md`, `flashinfer/experimental/qwen_3_5_block/_cudnn_frost/QSA/`)**:
   - **Qwen 3.8 Flash-Next Gated Sparse Attention (QSA) Decode Block on Rubin (SM107)**: Uses cuDNN Frontend `f2aa360e5` (FROST) to fuse QKV+Gate+Indexer projections, 4-token compressed-key cache writes, paged scoring, and decode top-$k$. At `seq_len = 32,768`, `top_k_blocks = 512` (`2,048` sparse tokens), and `d_head = 256`, HBM traffic drops from `33,554,432 B` to **`4,194,304 B` (`8.0x` bandwidth speedup)**.
5. **`NVIDIA/cutlass@0efeafb` (`#3424`), `dc45f97` (`#3556`), `957eb92` & `tile-ai/tilelang@efa1ffc` (`#3643`), `d85a70a` (`#3582`), `d9a31dd` (`#3579`)**:
   - TileLang's `ThreadPartialSyncRewriter` pre-populates `used_barrier_ids_` across `"shared"` and `"shared.dyn"` passes so disjoint thread groups never share the same `bar.sync id, count` named barrier (reducing barrier ID collisions from `2` to **`0`** and fixing hangs in warp-specialized sparse-MLA forward kernels).

## 2. Architect Design Exercises & Falsifiable Questions

- **Calculation Exercise**: Verify that at `seq_len = 32,768`, `compress_stride = 4`, `top_k_blocks = 512`, and `d_head = 256` (BF16), scanning `8,192` 128-dim BF16 compressed indexer keys (`2 MiB`) plus reading `2,048` selected K/V rows (`2 MiB`) totals `4 MiB` vs `32 MiB` dense K/V (`8.0x` bandwidth reduction).
- **Falsifiable Hypothesis**: Enabling `cluster_shape_m = 2` with `tile_size = 128` and `mma_inst_m = 128` (`atom_cta_m = 1`) on Rubin MoE causes a cluster to span two distinct routing tiles, risking out-of-bounds metadata reads or barrier hangs when `num_non_exiting_tiles` is odd (falsified and blocked by `rubin_moe_cluster_covers_one_tile == False`).
