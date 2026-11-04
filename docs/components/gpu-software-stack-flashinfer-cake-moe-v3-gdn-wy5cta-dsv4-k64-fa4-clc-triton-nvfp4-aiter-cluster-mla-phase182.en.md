# GPU Software Stack: FlashInfer Cake Mega-MoE v3 Single-FC2-Task, 5-CTA/SM GDN WY & SM107a `K=64` DSV4 Decode, FlashAttention-4 2CTA MLA Head Padding & SM100 Causal CLC, Triton NVFP4 SMEM Accounting & AITER `gfx1250` Cluster Split-K MLA (Phase 182)

> 🌐 **Language / 语言切换**: **[中文](gpu-software-stack-flashinfer-cake-moe-v3-gdn-wy5cta-dsv4-k64-fa4-clc-triton-nvfp4-aiter-cluster-mla-phase182.md)** | **[English](gpu-software-stack-flashinfer-cake-moe-v3-gdn-wy5cta-dsv4-k64-fa4-clc-triton-nvfp4-aiter-cluster-mla-phase182.en.md)**

---

## 1. Core Architectural Problems & Primary Source Evidence (Pillar 0 & Pillar 1)

Phase 182 (`priority: 191`) audits **13 upstream mainline commits** across Blackwell (`sm_100a` / B200 / GB200), Blackwell Ultra (`sm_103a` / GB300 / B300), Rubin (`sm_107a` / R200 / VR200 / GR100), AMD `gfx1250` (MI455X), and CDNA4 (`gfx950` / MI350X / MI355X) kernel and compiler stacks:

1. **FlashInfer `cake_mega_moe_v3` Single-FC2-Task TP8/EP1 Programs, 5-CTA/SM Packed-Triangular GDN WY Forward & KCP Tape Handoff**:
   - `1ce8f915659e` (#8669) optimizes `cake_mega_moe_v3` `TP8 EP1` (`T=32 / 128 / 512 / 1024`) rank programs on GB300 (`sm_103a`, 152 SMs) for 384-expert top-6 `hidden=5120` models so each FC2 ring task covers all `4` N-clusters of a claimed expert group (`1` FC2 task/group instead of `2`), achieving **`1.010x`** over previous Cake at `T=32` (`91.26 -> 90.36 us`) and `T=128` (`176.58 -> 174.97 us`), and beating autotuned `trtllm_fp4_block_scale_routed_moe` by **`1.021x–1.174x`** across all 32 rank shapes (`1.173x` at `TP8 EP1 T=32`: `105.88 -> 90.27 us`; **`1.174x`** at `TP8 EP8 T=4096`: `209.05 -> 178.14 us`).
   - `8e7be5e3a832` (#8665) adds a 5-CTA/SM packed-triangular forward WY program (`wy_tc2`) in `cake_gdn_train` on B200 (`sm_100a`) and GB300 (`sm_103a`), delivering up to **`2.013x`** `fwd+bwd` speedup over cuDNN FROST (chunk 64) on GB300 (`1.2787 -> 0.6351 ms`) and up to **`2.527x`** on 212-SM R200 (`sm_107a`, `0.6136 -> 0.2428 ms`).
   - `283abf0d4eab` (#8638) hands recompute tapes directly to backward during `graph="replay"` checkpointed steps in `cake_gdn_train_kcp` (skipping redundant `prep` and state walks), achieving **`1.249x`** on B200 (`6.050 -> 4.845 ms`), **`1.248x`** on GB300 (`5.598 -> 4.495 ms`), and **`1.257x`** on R200 (`3.554 -> 2.828 ms`) at 32K tokens.
2. **FlashInfer SM107a `K=64` FP8 DSV4 Decode, Bound-Launch Dense Projection GEMM & SM103 Cross-Band QKV Exchange**:
   - `a8e34e486334` (#8668) adds a single-CTA `M128`, `K=64` FP8 MMA program for the 128-head uniform-gather persistent DeepSeek-V4 decode route on VR200 (`sm_107a`, 208 SMs) when `rows >= 48`, reaching up to **`1.2778x`** speedup (`14.623 -> 11.444 us` at `rows=124, top-k 128x (384)`) with bitwise-identical outputs.
   - `02fa872444d6` (#8650) routes `cake_dense_projection_gemm` non-deterministic entry points directly from their cache key to the bound launch (skipping re-planning and DLPack conversion), accelerating host-bound eager backward and checkpointed steps by **`1.068x–1.136x`** on GB200 (`4096->128` bwd `56.58 -> 50.10 us` = **`1.129x`**) and **`1.082x–1.129x`** on GB300 (`4096->32` bwd `38.42 -> 34.03 us` = **`1.129x`**).
   - `7d929987724c` (#8649) adds hidden-size per-destination thresholds (`8192 / 16384 / 32768` rows) and an SM103 8-rank cross-band (`_xb`, `M >= 4096`) persistent launch plan for `QkvNormRopeExchangeOverlap`.
3. **FlashAttention-4 Dense 2CTA MLA Head Padding & SM100 Causal CLC + Global LPT Defaults**:
   - `5f7b05a78bf0` (#3118) pads `65–127` query heads in dense 2CTA absorbed-MLA forward (such as 96-head Kimi-K3 at TP=1) to the 128-row cluster tile in-kernel via `qheads_first_tma_view`, accelerating B200 decode at `batch=512, s_k=256` from `152.4 us` (1CTA) to **`68.5 us` (`2.225x`)** and `batch=64, s_k=8192` from `273.5 us` to **`169.6 us` (`1.613x`)**.
   - `6d6392ddc66d` (#3062) enables SM100/SM103 causal forward on CLC by default with late load-warp claiming (`clc_late_claim`) and Global LPT ordering at `<= 2.25` tiles/SM, delivering **`1.07x–1.28x`** on B200, **`1.05x–1.23x`** on B300, and **`1.04x–1.41x`** on GB300 partial-section grids, plus **`1.006x–1.148x`** (geomean `1.044x`) for `hd256` 2CTA on B200 (`>= 32` slices).
4. **Triton Decoded NVFP4 SMEM Accounting, TileLang SM103a/SM107a TMA Gates & ROCm AITER `gfx1250` Cluster Split-K MLA / `gfx950` Q256 MXFP4**:
   - `90e50ecd9cb4` (#12613) accounts for decoded BF16 weight tiles and regular-kernel conversion scratch in Blackwell `FP8 x NVFP4` matmul so configs stay within GB300's `232,448 B` SMEM limit (down from `236,840 B` / `266,256 B`), reducing `M=4096, K=8192, N=16384` latency from `2,711.89 us` to **`2,057.72 us` (`1.318x`, `-24.1%`)**; `51f4f818afc1` (#12524) enables LDS prefetch for `gfx1250` `scaled_dot`; TileLang `8587866b22c1` (#3772) accepts `sm_103a` (B300) and `sm_107a` (GR100/Rubin) in the CuTeDSL TMA arch gate.
   - ROCm AITER `0f61f2b95d0b` (#7390) folds `2/4/8/16` split-K reductions in-kernel via `(S, 1, 1)` workgroup clusters using LDS + TDM staging on `gfx1250` MLA v4.1 sparse prefill, eliminating `pa_decode_ps_reduce_hip_kernel` and cutting `H=64, N=8, topk=1024` latency from `15.26 us` to **`8.36 us` (`1.825x`, `-45.2%`)**; `0e072ca50ee7` (#5988) adds the `gfx950` Q256 fused MXFP4 prefill pipeline (in-assembly weighted M128 scheduling + `AsmMoeMode` mode 3 post-reduction), improving captured routes by **`5.40%–8.61%`** vs FlyDSL (`1,125.49 us` full-API graph median, max NRMS `0.00017653`).

---

## 2. End-to-End Dataflow & Architecture Topology (Pillar 1)

```
+--------------------------------------------------------------------------------------------------------+
| Phase 182 GPU Software Stack: Blackwell B200/GB300, Rubin R200/VR200 (sm_107a), gfx1250 & CDNA4 gfx950 |
+--------------------------------------------------------------------------------------------------------+
  [FlashInfer Cake Mega-MoE v3, GDN 5-CTA WY & DSV4 K=64 (#8669, #8665, #8638, #8668, #8650, #8649)]
    |-- Cake Mega-MoE v3 (#8669)     --> GB300 sm_103a TP8/EP1 1 FC2 Task/Group (4 N-Clusters, 1.174x vs TRT-LLM)
    |-- Cake GDN 5-CTA WY (#8665)    --> B200/GB300 wy_tc2 5 CTAs/SM (2.013x GB300 / 2.527x R200 vs FROST)
    |-- Cake GDN KCP Tape (#8638)    --> Graph-Replay Recompute Tape Handoff to Backward (1.257x R200 @ 32K)
    |-- Cake DSV4 SM107a (#8668)     --> 128-Head Uniform-Gather K=64 FP8 MMA rows>=48 (1.2778x VR200)
    +-- Bound Proj GEMM (#8650/#8649)--> Key-to-Bound-Launch Eager Bypass (1.129x) + SM103 8-Rank XBand QKV
                                      |
                                      v
  [FlashAttention-4 + Triton + TileLang + ROCm AITER (#3118, #3062, #12613, #12524, #3772, #7390, #5988)]
    |-- FA4 Dense 2CTA MLA (#3118)   --> In-Kernel TMA Pad 65-127 Q-Heads to 128 (2.225x B200 96-Head Decode)
    |-- FA4 SM100 Causal CLC (#3062) --> Default CLC + Late Load Claim + Global LPT <=2.25 Tiles/SM (1.41x GB300)
    |-- Triton & TileLang (#12613)   --> GB300 Decoded NVFP4 SMEM <=232,448B (1.318x) + SM103a/SM107a TMA Gate
    +-- ROCm AITER (#7390, #5988)    --> gfx1250 Cluster Split-K MLA (1.825x) + gfx950 Q256 MXFP4 (+8.61%)
```

---

## 3. Deterministic Offline Verification (Pillar 2 & Pillar 3)

Verified via [`experiments/flashinfer_cake_moe_v3_gdn_wy5cta_fa4_clc_aiter_cluster_mla_phase182/run.py`](../../experiments/flashinfer_cake_moe_v3_gdn_wy5cta_fa4_clc_aiter_cluster_mla_phase182/run.py) `--check`:
- `cake_mega_moe_v3_t32_self_speedup = 1.010`, `cake_mega_moe_v3_vs_trtllm_max_speedup = 1.174` (`1` FC2 task/group)
- `cake_gdn_5cta_gb300_vs_frost_speedup = 2.013`, `cake_gdn_r200_vs_frost_max_speedup = 2.527`, `cake_gdn_kcp_tape_handoff_r200_speedup = 1.257`
- `cake_dsv4_sm107a_k64_max_speedup = 1.2778` (`rows >= 48` threshold), `cake_proj_gemm_bound_launch_gb200_speedup = 1.129`
- `fa_mla_2cta_dense_96h_decode_speedup = 2.225`, `fa_sm100_clc_global_lpt_gb300_max_speedup = 1.41` (`<= 2.25` tiles/SM)
- `triton_gb300_fp8xnvfp4_speedup = 1.318` (within `232,448 B` SMEM budget), `tilelang_cutedsl_new_tma_archs = 2`
- `aiter_gfx1250_mla_cluster_splitk_speedup = 1.825`, `aiter_gfx950_q256_mxfp4_vs_flydsl_max_gain_pct = 8.61`

---

## 4. Enterprise Business Scenario Playbook (Pillar 4)

| Scenario Dimension | ✅ Good Fit | ⛔ Not For |
| :--- | :--- | :--- |
| **GB300 / R200 Mega-MoE & GDN Hybrid Training/Inference** | Enable `cake_mega_moe_v3` single-FC2-task programs on GB300 `sm_103a` `TP8 EP1` (`#8669`, `1.174x` vs TRT-LLM), 5-CTA/SM `wy_tc2` GDN WY (`#8665`, `2.013x`), and KCP graph-replay tape handoff (`#8638`, `1.257x`). | Forcing `K=64` uniform-gather DSV4 decode on VR200 (`sm_107a`) when `rows < 48` (back-to-back launches are slower than `K=32` below 48 rows). |
| **Blackwell 96-Head Dense MLA & Causal Attention Scheduling** | Use FA4 2CTA in-kernel head padding for `65–127` query-head dense absorbed MLA such as Kimi-K3 TP=1 (`#3118`, `2.225x` decode speedup) and SM100 CLC + Global LPT at `<= 2.25` tiles/SM (`#3062`, up to `1.41x` on GB300). | Padding `< 65` dense MQA heads to 128 on 2CTA (3 heads is `18x` slower than 1CTA) or enabling Global LPT above `2.25` tiles/SM (`1.8%–8%` loss on full-section grids). |
| **Blackwell `FP8 x NVFP4` Matmul & AMD `gfx1250`/`gfx950` Sparse/MoE Prefill** | Use Triton 3.9's corrected decoded-NVFP4 SMEM accounting on GB300 (`#12613`, `1.318x`), `gfx1250` in-kernel cluster split-K MLA v4.1 (`#7390`, `-45.2%`), and `gfx950` Q256 fused MXFP4 prefill (`#5988`). | Dispatching AITER Q256 MXFP4 on unshuffled or interleaved weights (must enforce both-projection shuffle flags and retain two-stage FP8 fallback). |
