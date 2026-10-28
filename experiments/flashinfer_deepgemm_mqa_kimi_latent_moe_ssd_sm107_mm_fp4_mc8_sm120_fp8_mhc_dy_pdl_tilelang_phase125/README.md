# Phase 125 Track A: FlashInfer DeepGEMM MQA, Kimi-K3 LatentMoE, SM107 SSD, GB300 Cluster-of-8 FP4, SM120 FP8, mHC dY Release, PDL Barrier & TileLang

Deterministic CPU/offline analytical suite (`own-offline`) modeling:
1. **FlashInfer Cake DeepGEMM Dense MQA SM100 (`#7741`, `512db9b3`)**: Serving `fp8:h64:full:qle2048:kvsplit` with the partial-block program (`1.227x` mean speedup, `-1,396` LOC).
2. **FlashInfer Kimi-K3 LatentMoE L2 Prefetch (`#7742`, `4158fddf`)**: Front/tail decode projection L2 `partial_prefetch` + `sb_norm` (`1.410x` peak speedup).
3. **FlashInfer SM107a SSD 1-CTA Cluster & GB300 Cluster-of-8 Multicast FP4 GEMM (`#7740`, `ecd4d71d`; `#7739`, `c9fc6cab`)**: `__cluster_dims__(1,1,1)` across 56 SSD programs (`1.246x`) and `mc42` 2x2 pair-tile multicast (`1.363x` mean speedup, `-50%` HBM operand reads).
4. **FlashInfer SM120 Weight-First FP8 & mHC Backward `dY` Release (`#7520`, `1e7a60d2`; `#7718`, `1fb63709`)**: `1.184x` grouped TC overlap and releasing `508.2 MiB` per mHC connection (`15,754.2 MiB` peak memory saved across 32 connections).
5. **MoE EP Pre-Dispatch Quant, SM90 GDP Varlen, PDL Wait Synchronization, Prims-TS Rubin Locality & TileLang (`#7426`, `#7713`, `#6322`, `#6832`, `#3374`, `#3367`, `#3341`, `#3366`)**: `-71.88%` NVLink payload via pre-dispatch NVFP4 quantization, `0.0` PDL scale race rate, `1.265x` Rubin 2-way locality-domain MoE speedup, and `0` TileLang batched all-reduce OOB threads.

## Run
```bash
python experiments/flashinfer_deepgemm_mqa_kimi_latent_moe_ssd_sm107_mm_fp4_mc8_sm120_fp8_mhc_dy_pdl_tilelang_phase125/run.py
python experiments/flashinfer_deepgemm_mqa_kimi_latent_moe_ssd_sm107_mm_fp4_mc8_sm120_fp8_mhc_dy_pdl_tilelang_phase125/run.py --check
```
