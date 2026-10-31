# Phase 156 Deep Dive: FlashInfer SM103a MegaMoE v3 W4A8, mHC Training Kernel, KDA M64 Owner/Helper, Ulysses NVLink Vectorization, NVFP4 Sparse MLA & Triton/TileLang Lowering

> **Language / 语言**: [English](flashinfer-megamoe-sm103a-mhc-kda-ulysses-triton-tilelang-phase156.en.md) | [中文](flashinfer-megamoe-sm103a-mhc-kda-ulysses-triton-tilelang-phase156.md)

## 1. Overview & Architectural Evolution

This report audits 10 primary commits across `flashinfer-ai/flashinfer`, `triton-lang/triton`, and `tile-ai/tilelang` focusing on Blackwell Ultra (`sm_103a`) W4A8 MoE pipelines, Multi-Stream Hyper-Connections (`mHC`) training kernels, KDA recurrent state scheduling, NVFP4 sparse MLA caches, and compiler occupancy/type-conversion lowering.

## 2. Primary Source Audit & Quantitative Findings

1. **FlashInfer `cake_mega_moe_v3` SM103a TP4/TP8 & EP-Local Routing (`10bd87c`, `#7633`)**: Adds `sm_103a` kernels for DeepSeek V4.1 Flash W4A8 (`e384_i384`) with EP-local rank shortcuts, reducing TP8 dispatch/combine latency from `148.0 us` to `82.88 us` (**1.79x** speedup).
2. **FlashInfer `cake_mhc_train` Stream-Mean (`bc5093b`, `#7856`) & KDA M64 Owner/Helper (`d9fe063`, `#7866`)**: Fuses forward/backward stream-mean reduction into an autograd node (**3.0x** HBM traffic reduction) and accelerates 36-task M64 FP32-state KDA prefill by **1.60x** (`135.0 us` vs `216.0 us`).
3. **FlashInfer Portable Ulysses NVLink A2A (`82e9362`, `#7015`) & NVFP4 Sparse MLA (`1259b44`, `#7410`)**: Achieves **88.0%** NVLink bandwidth efficiency (**792 GB/s**, **1.55x** over scalar) and compresses MLA KV caches by **3.27x** (`352 B/token` vs `1152 B/token`).
4. **Triton Non-Aligned Warp Barriers (`2cd5519`, `#12321`), Driver SMEM Reservation (`4a18983`, `#12326`) & TileLang (`#3308`, `#3204`, `#3246`)**: Accounts for `1024 B` CUDA driver SMEM reservation to preserve **2 CTAs/SM**, adds the Ascend 950 backend, fuses exact `FP4 -> FP32 -> FP8` conversion (**2.80x** faster), and clamps `coalesced_width` safely instead of crashing with `LOG(FATAL)`.

## 3. Reproducible Verification

Run `python experiments/flashinfer_megamoe_sm103a_mhc_triton_barriers_tilelang_fp4_phase156/run.py --check`.
