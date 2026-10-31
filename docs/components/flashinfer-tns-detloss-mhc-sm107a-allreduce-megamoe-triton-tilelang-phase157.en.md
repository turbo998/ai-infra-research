# Phase 157 Deep Dive: FlashInfer SM100a Tail-N-Split GEMM, Deterministic Loss dX K-Slice Clamp, SM107a (R200) Fast Coefficient Chain, SM103a Fence-Warp AllReduce & MegaMoE v3 Preemption Fix, Triton ConSan Cluster Sync & TileLang Lowering

> **Language / 语言**: [English](flashinfer-tns-detloss-mhc-sm107a-allreduce-megamoe-triton-tilelang-phase157.en.md) | [中文](flashinfer-tns-detloss-mhc-sm107a-allreduce-megamoe-triton-tilelang-phase157.md)

## 1. Overview & Architectural Evolution

This report audits 10 primary commits across `flashinfer-ai/flashinfer`, `triton-lang/triton`, and `tile-ai/tilelang` covering Blackwell (`sm_100a`), Blackwell Ultra (`sm_103a`), and Vera Rubin (`sm_107a` / R200) tensor kernels alongside Triton ConSan multi-CTA synchronization and TileLang JIT/Autotuner infrastructure.

## 2. Primary Source Audit & Quantitative Findings

1. **FlashInfer SM100a Tail-N-Split GEMM (`f6b47a5`, `#8111`) & `cake_lm_head_det_loss` K-Slice Clamp (`96ca3d0`, `#8108`)**: Splitting leftover 256-column pair tiles into 128-column halves on the final wave of Kimi K3 TP8 `o_proj` doubles active 2-CTA clusters (`22 -> 44`, `59.46%` utilization) and reduces $M=2560$ latency from `18.02 us` to `16.90 us` (**1.0663x** speedup). Clamping `k_slices` via `dx_slices_fit` prevents 0-step K-slices from skipping SMEM zero-init at small $V=256$, cutting relative L2 `dX` error vs FP64 from `0.682` to `2.565e-3` (**265.9x** reduction).
2. **FlashInfer SM107a (`R200`) Fast Coefficient Chain (`fec6b56`, `#8105`), SM103a Fence-Warp AllReduce (`83f6dc8`, `#8102`) & MegaMoE v3 Preemption Fix (`18fb0bf`, `#8095`)**: Accelerates `sm_107a` UE8M0/NVFP4 mHC tailpipes to `13.888 us` (**1.0363x** on 208-SM R200) with bitwise-identical RN division, cuts SM103a 1-token BF16 `world_size=2` AllReduce to `7.49 us` (**1.0320x**), and eliminates GPU preemption hangs (**100% -> 0%**) by zeroing `task_infos` shared memory before barrier synchronization.
3. **Triton ConSan Cluster Sync (`1649179`, `#11908`), Unsigned Pointer Subtraction (`6706b88`, `#12318`) & TileLang (`#3337`, `#3447`, `#3450`)**: Synchronizes `0x7fffffff` poison initialization across CTAs via cluster barriers, widens `<64-bit` unsigned pointer offsets before negation (preventing **256-element** wraparound on `uint8`), forwards `@autotune` profiler args, emits AscendC `#line` directives, and supports disk-cached TVM `Module` objects.

## 3. Reproducible Verification

Run `python experiments/flashinfer_tns_detloss_mhc_sm107a_allreduce_triton_tilelang_phase157/run.py --check`.
