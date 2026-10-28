# FlashInfer Cake NVFP4 B200 `M=1` PDL Chain & GB300 Folded-Scale Multicast GEMM, Blackwell Softmax Routing, Causal Conv1D Stage-Pair Bwd, GDN Adjoint-Walk Tape, Frost MX Cross-Arch Loader, and Triton MMAv5 `warpid=0` Fold / CDNA3-4 Buffer Atomics (Phase 131 · Priority 140)

> **中文版**: [flashinfer-cake-fp4-conv-gdn-triton-warp-fold-cdna-phase131.md](flashinfer-cake-fp4-conv-gdn-triton-warp-fold-cdna-phase131.md)  
> **Evidence Anchor**: `ev-flashinfer-cake-fp4-conv-gdn-triton-warp-fold-cdna-phase131-2026` (`own-offline` + `implementation`)  
> **Reproducible Experiment**: `python experiments/flashinfer_cake_fp4_conv_gdn_triton_warp_fold_cdna_phase131/run.py --check`

---

## 1. Primary Source Matrix

- **`flashinfer-ai/flashinfer` (`449a357`, `59f1b2c`, `500833a`, `ff078b4` & `bdb35ef`)**: B200 `M=1` PDL `quantize -> GEMM` chain (**`18.4 -> 12.6 us`, `1.4603x`**) and GB300 multicast GEMM with folded `out_scale` at `M=8192, K=28672` (saving **`234,881,024 B` / `224.0 MiB` HBM**, **`412.0 -> 338.5 us`, `1.2171x`**); Blackwell Softmax batch-window dispatcher routing; `cake_causal_conv1d_train` stage-pair backward fallback for `C < 512` (**`94.0 -> 71.5 us`, `1.3147x`**); context-parallel GDN training (`H_qk=16, HV=48`, `3:1` FP32 value-head reduction) forward prep and adjoint-walk tape kernels (**`168.0 -> 128.0 us`, `1.3125x`**); and Frost MX cross-architecture TVM-FFI module materialization (**`100% -> 0.0%` uninitialized FFI failure**).
- **`triton-lang/triton` (`1b4a204`, `a950b39`, `71114a1`, `dc1d703` & `cb0f110`)**: Folding relative `warpid` to `0` for single-warp partitions in `ConvertWarpSpecializeToLLVM.cpp` eliminates **`128 -> 0` warp-id conditional branches** around `tcgen05.mma` per 128-tile loop; caching CUDA device properties cuts repeated driver queries **`1000 -> 1`**; disabling unsupported AMD multi-CTA scalar atomics and adding `cdna{3,4}.buffer.atomic_load/store` (`llvm.amdgcn.raw.ptr.atomic.buffer.load`) prevents LLVM from hoisting spin-wait flag polls (**`1 -> 0` poll deadlocks**).
