# Phase 182: GPU Software Stack Deterministic Verification Suite

Validates the quantitative kernel & compiler claims for Phase 182 (`priority: 191`):
- **FlashInfer `#8669` (`1ce8f915659e`)**: `cake_mega_moe_v3` single-FC2-task/group (`4` N-clusters) TP8/EP1 programs on GB300 (`sm_103a`, `1.010x` vs previous Cake at T32/T128, up to `1.174x` vs autotuned `trtllm_fp4_block_scale_routed_moe`).
- **FlashInfer `#8665` (`8e7be5e3a832`), `#8638` (`283abf0d4eab`), `#8668` (`a8e34e486334`) & `#8650` (`02fa872444d6`)**: `cake_gdn` 5-CTA/SM packed-triangular forward WY (`2.013x` on GB300 and `2.527x` on R200 vs cuDNN FROST chunk-64), KCP graph-replay tape handoff (`1.257x` on R200 at 32K tokens), SM107a 128-head uniform-gather `K=64` FP8 DSV4 decode (`1.2778x` at `rows=124`, active for `rows >= 48`), and bound-launch projection GEMM (`1.129x` on GB200).
- **FlashAttention `#3118` (`5f7b05a78bf0`) & `#3062` (`6d6392ddc66d`)**: Dense 2CTA absorbed-MLA head padding (`65–127 -> 128` rows, `2.225x` on B200 96-head decode) and SM100 causal CLC + Global LPT (`1.41x` max on GB300 partial sections at `<= 2.25` tiles/SM).
- **Triton `#12613` (`90e50ecd9cb4`), `#12524` (`51f4f818afc1`), TileLang `#3772` (`8587866b22c1`), and ROCm AITER `#7390` (`0f61f2b95d0b`) & `#5988` (`0e072ca50ee7`)**: Blackwell `FP8 x NVFP4` decoded-tile SMEM accounting (`1.318x` on GB300 within `232,448 B`), `gfx1250` MLA v4.1 in-kernel cluster split-K (`1.825x`, `-45.2%` at `H=64, N=8, topk=1024`), and `gfx950` Q256 fused MXFP4 prefill (`+8.61%` vs FlyDSL).

## Run
```bash
python experiments/flashinfer_cake_moe_v3_gdn_wy5cta_fa4_clc_aiter_cluster_mla_phase182/run.py --check
```
