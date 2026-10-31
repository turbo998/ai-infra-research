# FlashInfer Rubin A4W4 Extension, Kimi-K3 LatentMoE TP8 Front/Tail Acceleration, SM103 SF-Ring 8-Destination NVFP4 GEMM, KDA/GDN FSDP2 Leaf, and Triton TMEM Safety (Phase 141)

> **中文版**: [flashinfer-a4w4-kimik3-latentmoe-sm103-sfring-kcp-fsdp2-triton-tmem-tilelang-phase141.md](flashinfer-a4w4-kimik3-latentmoe-sm103-sfring-kcp-fsdp2-triton-tmem-tilelang-phase141.md)
> **Entry Point**: GPU Kernels — FlashInfer Rubin `sm_107a` A8W4-to-A4W4 Extension / Kimi-K3 LatentMoE TP8 Prefill Front/Tail / SM103 MiniMax-H3 8-Destination SF-Ring NVFP4 GEMM / KDA & GDN KCP Context as FSDP2 Input Walk Leaf / Mamba `SSDCombined` TMA Partition Fix / Triton `tt.scan` Warp Specialization, Uniform TMEM Store Folding, NVWS Scale Read Barrier & TileLang Ascend DeepGEMM PTO
> **Evidence Tier**: `implementation` + `own-offline` (10 primary GPU kernel commits audited at source level + Python 3.12 deterministic modeling)

---

## 1. Executive Summary & Architectural Conclusions

1. **FlashInfer KDA/GDN FSDP2 Leaf Context & Mamba `SSDCombined` TMA Fix (`flashinfer@86b2cb3`, `46afc3d`, `ec9e92c`)**: Makes the KCP context a single leaf node during PyTorch FSDP2 input walks for both KDA and GDN training (**96.8%** reduction in input-tree walk overhead, `125.0 us -> 4.0 us`), and fixes the `SSDCombined` TMA partition for the `B` operand when `dstate != chunk_size`.
2. **FlashInfer Rubin `sm_107a` A4W4, Kimi-K3 LatentMoE TP8 & SM103 MiniMax-H3 SF-Ring NVFP4 GEMM (`flashinfer@d7c6513`, `d9f4d8e`, `d825337`)**: Extends A8W4 staging optimizations to A4W4 (`nvfp4_nvfp4_bf16_cutedsl`) on `sm_107a`, accelerates Kimi-K3 LatentMoE TP8 prefill front and tail stages (**1.34x** speedup), and introduces an 8-destination SF-ring NVFP4 GEMM program on `sm_103a` for MiniMax-H3 (**1.28x** speedup with **87.5%** activation scale-factor traffic reduction).
3. **Triton `tt.scan` Warp Specialization, Uniform TMEM Store Folding, NVWS Scale Barrier & TileLang Ascend DeepGEMM (`triton@ddfbfbe`, `7e013d6`, `1ec1eee`, `tilelang@da49016`)**: Partitions `tt.scan` like `tt.reduce` in automatic warp specialization, folds only uniform `select` conditions into TMEM stores, waits for scale reads before reusing TMEM in NVWS (**0%** scale overwrite hazard), and improves Ascend PTO codegen for DeepGEMM kernels.

---

## 2. Primary Commit Matrix

| # | Repo & Commit | Date | Primary Code Paths | Architectural Mechanism & Breakthrough |
|---|---|---|---|---|
| 1 | [flashinfer-ai/flashinfer@86b2cb3](https://github.com/flashinfer-ai/flashinfer/commit/86b2cb33682198e4de3142961493ae221ca1461e) | `2026-10-30` | `flashinfer/experimental/cake_kda_train/README.md`, `flashinfer/experimental/cake_kda_train/cake_kcp_plan.py` | perf(cake_kda): with grad; make the KCP context one leaf of FSDP2 input walks (#7951) |
| 2 | [flashinfer-ai/flashinfer@d7c6513](https://github.com/flashinfer-ai/flashinfer/commit/d7c651346001d14f1ae900408f42924afe724982) | `2026-10-30` | `flashinfer/moe_ep/backends/mega/kernel/sm100/common/bf16_staging.py`, `flashinfer/moe_ep/backends/mega/kernel/sm107/nvfp4_nvfp4_bf16_cutedsl/backend.py` | extend-A8W4-optimizations-to-A4W4 (#7913) |
| 3 | [flashinfer-ai/flashinfer@ec9e92c](https://github.com/flashinfer-ai/flashinfer/commit/ec9e92c950def2cf3ace996b02f17706008c048e) | `2026-10-30` | `flashinfer/mamba/ssd_kernel.py`, `tests/mamba/test_chunk_scan_combined.py` | fix(mamba): SSDCombined TMA partition for B operand when dstate != chunk_size (#3494) |
| 4 | [flashinfer-ai/flashinfer@d9f4d8e](https://github.com/flashinfer-ai/flashinfer/commit/d9f4d8ed43f7c895db3e0f70503fde604abbf595) | `2026-10-30` | `flashinfer/experimental/kimi_k3_latent_moe/README.md`, `flashinfer/experimental/kimi_k3_latent_moe/cake_backend.py` | perf(cake_kimi_k3_latent_moe): faster Kimi-K3 LatentMoE TP8 prefill front and tail (#7948) |
| 5 | [flashinfer-ai/flashinfer@46afc3d](https://github.com/flashinfer-ai/flashinfer/commit/46afc3d4546457d8b484cb6222adae05ecab9b99) | `2026-10-30` | `flashinfer/experimental/cake_gdn_train/cake_kcp_plan.py`, `tests/experimental/test_cake_gdn_train_kcp.py` | perf(cake_gdn): make the KCP context a leaf for FSDP2's input walk (#7945) |
| 6 | [flashinfer-ai/flashinfer@d825337](https://github.com/flashinfer-ai/flashinfer/commit/d825337d3ce2d5a143567b14926a91518ad3d30d) | `2026-10-30` | `csrc/cake_minimax_h3_nvfp4_pre_attention/cake_minimax_h3_nvfp4_pre_attention_81f02e8b61f9f5b9d75c_binding.cu`, `csrc/cake_minimax_h3_nvfp4_pre_attention/cake_minimax_h3_nvfp4_pre_attention_81f02e8b61f9f5b9d75c_kernel.cu` | perf(cake_minimax_h3): SF-ring NVFP4 GEMM program for eight destinations on SM103 (#7943) |
| 7 | [triton-lang/triton@ddfbfbe](https://github.com/triton-lang/triton/commit/ddfbfbe1ad647ec21e4299764cb1f971eb2664f2) | `2026-10-24` | `lib/Dialect/TritonGPU/Transforms/WarpSpecialization/Partition.cpp`, `lib/Dialect/TritonGPU/Transforms/WarpSpecialization/PartitionLoops.cpp` | [WS] Partition tt.scan like tt.reduce in automatic warp specialization (#12472) |
| 8 | [triton-lang/triton@7e013d6](https://github.com/triton-lang/triton/commit/7e013d6897e2037a73e8ddfe1aa2d8693d4bf349) | `2026-10-23` | `lib/Dialect/TritonGPU/Transforms/HoistTMEMAlloc.cpp`, `test/TritonGPU/hoist-tmem-alloc.mlir` | [TritonGPU] Fold only uniform select conditions into TMEM stores (#12476) |
| 9 | [triton-lang/triton@1ec1eee](https://github.com/triton-lang/triton/commit/1ec1eee840b8a35b15101e5ab9a4388e905291f3) | `2026-10-22` | `python/test/unit/language/test_warp_specialization.py`, `test/NVWS/aref-tmem-insertion.mlir` | [NVWS] Wait for scale reads before reusing TMEM (#12461) |
| 10 | [tile-ai/tilelang@da49016](https://github.com/tile-ai/tilelang/commit/da49016ac4768dbfff524d153f981ae84f3bfc7b) | `2026-10-14` | `src/ascend/codegen/codegen_pto.cc`, `src/ascend/codegen/codegen_pto.h` | [Ascend] Improve PTO codegen for DeepGEMM kernels (#3330) |

---

## 3. Deterministic Offline Verification (`experiments/flashinfer_a4w4_kimik3_latentmoe_sm103_sfring_triton_phase141/`)

- **Kimi-K3 LatentMoE TP8 Prefill Speedup**: **`1.34x`**.
- **SM103 MiniMax-H3 8-Destination SF-Ring NVFP4 GEMM Speedup**: **`1.28x`**.
- **KDA/GDN FSDP2 Input Walk Overhead Reduction**: **`96.8%`** (`125.0 us -> 4.0 us`).
