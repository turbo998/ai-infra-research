# FlashInfer Cake SM90 INT8/FP8 Attention, tcgen05 Narrow Unroll & FP4 AllReduce Deslop (Phase 100)

Deterministic Python 3.12 standard-library model reproducing:
1. **FlashInfer `#7373` (`14165a5bd9`)**: Cake SM90 INT8 QK (with K-mean smoothing) + FP8 E4M3 PV 4-kernel chain (`statistics_partials` -> `statistics_finalize` -> `quantization` -> `attention`) across shapes `R1..R7` (`1.08x-1.16x` speedup over FlashAttention-3 BF16 on `R2-R7`, `0.86x` on `R1`).
2. **FlashInfer `#7370` (`e177fd16c9`) & `#7366` (`5b79a1f222`)**: `cake_tinygemm2_tcgen05` narrow-tile multi-round unrolling (`2` rounds on `sm_100a`, `4` rounds on `sm_103a`/`sm_107a`, `12 -> 20` programs) and `cake_fp4_allreduce` registry/binding consolidation (`116 -> 49` `.cu` files, `-81,881` net lines).
3. **FlashInfer `#6231` (`04430a391b`) & TileLang `#3523`/`#2887`**: Graph-mode sticky FA2 schedule refresh when `auto` re-plans, plus Ascend PTO `SIMTVectorLocalScalarOffset` (`slot * element_lanes`) and empty stepped-loop (`extent == 0 -> 0`) unrolling guard.
