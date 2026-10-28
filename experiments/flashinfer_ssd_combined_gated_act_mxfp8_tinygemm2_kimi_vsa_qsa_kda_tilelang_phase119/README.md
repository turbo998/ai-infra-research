# Phase 119 Track A: FlashInfer SM100/SM103/SM107 (R200) Kernels & TileLang Ascend Pipeline/Cache

Reproducible stdlib model covering 10 primary upstream commits across `flashinfer-ai/flashinfer` and `tile-ai/tilelang`:
- `#7558` (`372e9435f4`): `cake_ssd_combined` on `sm_107a` (Rubin R200) exact-scan (`1.012x–1.562x` across 44 changed shapes, `1.0487x` geomean across 93 shapes).
- `#7568` (`1d74db1e9d`): `cake_gated_act_mxfp8` 8-range `M*K` dispatch (`5` ranges on `sm_100a`, `3` on `sm_103a`, `0 B` spill packed `f32x2` sigmoid).
- `#7588` (`efa2600909`): `cake_tinygemm2_tcgen05` `CU_TENSOR_MAP_L2_PROMOTION_L2_256B` on `sm_107a` + packed-tile routing for `17..23` rows (`1.10x–1.46x`).
- `#7567` (`95ab86637c`): `cake_kimi_k3_fused_router` (`LCG_LW`, `Q4SB`, `Q4SPW_LR`, `1.05x` on GB300 & B200).
- `#7580` (`acce856763`): `cake_vsa_sm90` grouped-query FP8 queue & compact tile-boundary overlap (`1.0236x` geomean on 37 changed rows).
- `#7578` (`3055feed86`): `cake_qsa_nvfp4` direct partial epilogue on `sm_103a` (`1.03x`) & `sm_107a` (`1.06x`).
- `#7576` (`b17b12b2e5`): `cake_kda` `glm_intra_v18` gate-fused intra-chunk backward on `sm_107a` (`1.0481x` bwd).
- `#7583` (`8cf7089cf6`): `cake_trtllm_moe_allreduce_union` in-loop previous-epoch clear & 2-ahead reduce-scatter (`1.062x`).
- `#3624` (`dfe83e84ec`): TileLang Ascend manual pipeline configuration register phase preservation.
- `#3691` (`1c0f402e16`): TileLang Ascend device binary cache key with Bisheng build timestamp identity.
