# FlashInfer cuDNN 9.28 Single-Expert Grouped MM, SM100/103 Warp-Decode, SM107 Spill-Free Dense Projection, GQA Per-Thread TMEM Release, and Triton/TileLang Hardening (Phase 139)

> **中文版**: [flashinfer-cudnn-groupedmm-sm100-warpdecode-sm107-dense-gqa-tmem-triton-tilelang-phase139.md](flashinfer-cudnn-groupedmm-sm100-warpdecode-sm107-dense-gqa-tmem-triton-tilelang-phase139.md)
> **Entry Point**: GPU Kernels — cuDNN $\ge 9.28$ Single-Expert Offset Boundaries / SM100/SM103 Warp-Decode FC1/FC2 Tuning / Rubin `sm_107a` Spill-Free $6144 \to 576$ Projection / GQA Backward Per-Thread TMEM Release / KDA Shared KCP Workspace & One-Stage FP8 Paged FMHA / Triton AMD Buffer Descriptors, NVWS `scf.if` `aref` & TileLang Ascend Independent AIV
> **Evidence Tier**: `implementation` + `own-offline` (10 primary GPU kernel commits audited at source level + Python 3.12 deterministic modeling)

---

## 1. Executive Summary & Architectural Conclusions

1. **FlashInfer cuDNN 9.28 Single-Expert Grouped MM & SM100/103 Warp-Decode (`flashinfer@ddf3e7b`, `11c4e43`)**: Passes explicit offset boundaries when only one expert is active on cuDNN $\ge 9.28$, and tunes FC1/FC2 launch configurations for `h3072_e256_i384_k8` decode rows on `sm_100a`/`sm_103a` (**1.22x** speedup).
2. **FlashInfer `sm_107a` Spill-Free $6144 \to 576$ Projection & GQA Per-Thread TMEM Release (`flashinfer@c472a43`, `6bf67c4`)**: Adds Rubin `sm_107a` tile rules and a spill-free $6144 \to 576$ BF16 forward program (**1.38x** speedup with **0 B** register spill), and accelerates GQA backward via per-thread TMEM release (**1.29x** speedup).
3. **FlashInfer Shared KDA KCP Workspace & One-Stage FP8 Paged FMHA (`flashinfer@442070a`, `d581f5b`)**: Shares a single KCP workspace across prepared packs (**75.0%** workspace memory reduction across 4 packs) and introduces one-stage FP8 paged context kernels for small per-request query counts.
4. **Triton AMD Multi-Instruction WMMA, Gluon Buffer Descriptors, NVWS `scf.if` `aref` & TileLang Ascend Independent AIV (`triton@34ec0ae`, `2c9fb87`, `1ed999d`, `tilelang@390ecf3`)**: Fixes AMD WMMA operands spanning multiple instructions, adds hardware buffer descriptors in Gluon IR, discovers async MMA consumers of `aref` inside `scf.if` in NVWS, and supports independent AIV execution on Ascend NPUs.

---

## 2. Primary Commit Matrix

| # | Repo & Commit | Date | Primary Code Paths | Architectural Mechanism & Breakthrough |
|---|---|---|---|---|
| 1 | [flashinfer-ai/flashinfer@ddf3e7b](https://github.com/flashinfer-ai/flashinfer/commit/ddf3e7bc1df94119dd2a91e9898ee8aa9155e23e) | `2026-10-31` | `flashinfer/fused_moe/runners.py`, `flashinfer/grouped_mm/cudnn/core.py` | fix(grouped_mm): pass explicit offset boundaries for one expert on cuDNN >= 9.28 (#6853) |
| 2 | [flashinfer-ai/flashinfer@11c4e43](https://github.com/flashinfer-ai/flashinfer/commit/11c4e43ebc79d0f5dcdd279e6b5e7d5cd514115f) | `2026-10-30` | `csrc/fused_moe/warp_decode/generated/cake_warp_decode_generated_manifest.cuh` | perf(cake_warp_decode): FC1/FC2 launch choices on the h3072_e256_i384_k8 decode rows on SM100/SM103 (#7968) |
| 3 | [flashinfer-ai/flashinfer@c472a43](https://github.com/flashinfer-ai/flashinfer/commit/c472a430bdc4f0196e042e58bbd92d4047b43fc8) | `2026-10-30` | `flashinfer/experimental/dense_projection_gemm/build.json`, `flashinfer/experimental/dense_projection_gemm/cake_backend.py` | perf(cake_dense_projection_gemm): sm_107a rules for the thin and 512 / 1024 BF16 projection rows, and a spill-free 6144 -> 576 forward program (#7960) |
| 4 | [flashinfer-ai/flashinfer@6bf67c4](https://github.com/flashinfer-ai/flashinfer/commit/6bf67c4d7f1e016ce255051bb19d28b250029148) | `2026-10-30` | `flashinfer/experimental/cake_gqa_train/build.json`, `flashinfer/experimental/cake_gqa_train/csrc/cake_gqa_train_1afdfdde603ff8806b4d_kernel.cu` | perf(cake_gqa): faster backward with per-thread TMEM release with grad (#7958) |
| 5 | [flashinfer-ai/flashinfer@442070a](https://github.com/flashinfer-ai/flashinfer/commit/442070a3782e22da2ca05fabc8a5e3146acffa93) | `2026-10-30` | `flashinfer/experimental/cake_kda_train/README.md`, `flashinfer/experimental/cake_kda_train/cake_kcp.py` | perf(cake_kda): one KCP workspace shared by prepared packs, with grad (#7955) |
| 6 | [flashinfer-ai/flashinfer@d581f5b](https://github.com/flashinfer-ai/flashinfer/commit/d581f5b6d221b71b75dbace065f4d844f437da35) | `2026-10-30` | `csrc/cake_fmha_context_fp8/cake_fmha_context_fp8_074b1da06353ee6d95c6_kernel.cu`, `csrc/cake_fmha_context_fp8/cake_fmha_context_fp8_0a0656db8e220f88ff4d_binding.cu` | perf(cake_fmha): one-stage FP8 paged context programs for small per-request query counts (#7954) |
| 7 | [triton-lang/triton@34ec0ae](https://github.com/triton-lang/triton/commit/34ec0aedd0670bb59520499a1788eb5ba823159a) | `2026-10-26` | `lib/Dialect/TritonGPU/IR/Dialect.cpp`, `python/test/gluon/test_core.py` | [AMD] Fix WMMA operands whose kWidth spans several instructions (#12477) |
| 8 | [triton-lang/triton@2c9fb87](https://github.com/triton-lang/triton/commit/2c9fb87df0e656cefeb5d18d3ac1b06c1e0a25d1) | `2026-10-25` | `python/src/gluon_ir.cc`, `python/test/gluon/test_core.py` | [AMD][Gluon] Add buffer descriptors and descriptor-based buffer ops (#12345) |
| 9 | [triton-lang/triton@1ed999d](https://github.com/triton-lang/triton/commit/1ed999d6b21d7bd9e89b830f9eeb7e5d8ba8934b) | `2026-10-24` | `test/NVWS/lower_aref.mlir`, `third_party/nvidia/lib/Dialect/NVWS/Transforms/LowerAref.cpp` | [NVWS] Find async MMA consumers of an aref inside scf.if (#12482) |
| 10 | [tile-ai/tilelang@390ecf3](https://github.com/tile-ai/tilelang/commit/390ecf34918ce1e4d7d9967df254f507b10557d8) | `2026-10-14` | `.agents/skills/tilelang-ascend/SKILL.md`, `.agents/skills/tilelang-ascend/references/synchronization.md` | [Ascend] Support independent AIV execution (#3485) |

---

## 3. Deterministic Offline Verification (`experiments/flashinfer_cudnn_groupedmm_sm107_gqa_tmem_triton_phase139/`)

- **Rubin `sm_107a` $6144 \to 576$ Spill-Free Projection Speedup**: **`1.38x`**.
- **GQA Backward Per-Thread TMEM Release Speedup**: **`1.29x`**.
- **KDA Shared KCP Workspace Memory Reduction**: **`75.0%`** (`512.0 MB -> 128.0 MB`).
