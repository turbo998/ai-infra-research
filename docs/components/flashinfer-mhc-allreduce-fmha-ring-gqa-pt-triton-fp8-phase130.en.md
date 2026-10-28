# FlashInfer MNNVL mHC Fused AllReduce, GB300 4-Stage FP8 Context FMHA Ring, CuTe DSL 5-Stage Page-Table Ring Fix & Triton AMD Packed FP8e4nv Upcast (Phase 130)

> **Chinese Edition**: [flashinfer-mhc-allreduce-fmha-ring-gqa-pt-triton-fp8-phase130.md](flashinfer-mhc-allreduce-fmha-ring-gqa-pt-triton-fp8-phase130.md)  
> **Phase 130 (`priority: 139`) · 130-Phase & 400-Experiment-Suite Milestone · Track 1 (GPU Kernels & Compilers)**  
> **Evidence Class**: `implementation` + `author-benchmark` (first-hand upstream commits) + `own-offline` (deterministic CPU analytical modeling; not a GPU hardware rerun)

---

## 1. Architectural Context & Problem Statement

Across Blackwell (`sm_100a` B200, `sm_103a` GB300, `sm_120a` RTX 5090 / RTX PRO 6000), Rubin (`sm_107a` R200), Hopper (`sm_90a`), and AMD RDNA3/CDNA3/4 kernel stacks, 10 upstream commits landed between 2026-10-26 and 2026-10-28 expose five critical engineering inflection points:

1. **Launch & Synchronization Wall at Multi-Stream Hyper-Connection (mHC) Decode Boundaries**: GLM-5.3-Flash replaces the single residual stream with **4 mHC residual streams** (`hidden = 4096`) and 24 mix logits (`4 pre + 4 post + 16 comb`). Under TP8 on Blackwell MNNVL (e.g., two 4-GPU GB300 trays), every attention and MoE/MLP sublayer boundary—**89 boundaries per decode step**—must all-reduce across 8 TP ranks, mix into the 4 mHC streams (*post*), and compute the next sublayer's mHC *pre* partial projections. At decode batch sizes (`1..32` tokens), running 3 separate kernels per boundary (267 launches/step) is completely launch- and barrier-bound.
2. **GB300 (`sm_103a`) 8-Warp Input-Gradient Projection GEMMs & 4-Stage FP8 Context FMHA Ring**: While Cake's `dense_projection_gemm` defaults to a 10-warp pipeline across 126 programs, three BF16 `kn` input-gradient rows on `sm_103a` (`4096->512`, `5120->512`, and `kv_a 6144->576`) achieve lower latency with 8-warp CTAs (`w8rs`/`w8r`). Concurrently, GB300 (`sm_103a`) upgrades its exact FP8 paged context FMHA programs from a 3-stage to a **4-stage K/V ring** (`1.081x` host scale, `1.078x` device scale), while B200 (`sm_100a`) retains 3 stages.
3. **Plan-Dependent Producer/Consumer Overlap on SM120 (`RTX 5090` / `RTX PRO 6000`)**: Fixed overlap width `P = 32` in `minimax_h3_sm120_varlen_attention_fp8` underutilizes SMs when the attention-weighted mean segment length $L$ is either very short (producer-bound) or very long over large padded token counts.
4. **CuTe DSL Paged GQA Page-Table Ring Race (`pt_stages = 4 -> 5`)**: In `GroupedQueryAttentionDecodePaged`, the TMA K warp prefetches tile `s + 1` page indices into the shared-memory page-table ring before waiting on `tma_order_v_nbar`. Because the TMA V warp may still read slots `s - prefetch_iters - 1 .. s`, `prefetch_iters + 3 = 5` live slots are required at default `prefetch_iters = 2`. With `pt_stages = 4`, K-warp run-ahead overwrote slot `s` with tile `s + 4` page indices, causing **30%–50% relative error** whenever `q_len_per_req > 1` iterated $\ge 16$ tiles per CTA.
5. **Hopper VSA Two-Step Shuffle-Exchange & Triton AMD Packed `fp8e4nv -> fp16` Upcast**: Compact GQA block-sparse attention on `sm_90a` cuts `SHFL` instructions by 32 via two-step butterfly exchange, while Triton's AMD backend replaces per-element integer exponent add + 8-entry subnormal LUT with packed `i32` masking and a single `2^(15-7)` FP16 multiply (**up to 3.0x** RDNA3 GEMM speedup).

---

## 2. First-Hand Upstream Evidence Matrix (10 Commits)

| # | Repository & Commit | Date | Mechanism & Quantitative Architectural Impact |
|---|---|---|---|
| 1 | `flashinfer-ai/flashinfer@1af7b11` (`#7758`) | 2026-10-28 | Adds two 8-warp `sm_103a` programs (`dense_proj_gemm_kn_n256_tma1_w8rs` and `w8r`) to `cake_dense_projection_gemm` (126 programs / 123 templates). At $T=16231$, latency drops from `42.8 -> 41.2 µs` (**1.040x**) on `4096->512`, `52.2 -> 49.7 µs` (**1.050x**) on `5120->512`, and `67.1 -> 64.0 µs` (**1.048x**) on `kv_a (6144->576)`. |
| 2 | `flashinfer-ai/flashinfer@cb91a3e` (`#7752`) | 2026-10-28 | Upgrades `flashinfer.cake_fmha` (28 programs/target) exact FP8 context programs (`heads 32/4, page 64`) on GB300 (`sm_103a`) from a 3-stage to a **4-stage K/V ring** (**1.081x** faster with host scales, **1.078x** with device scales) and adds smaller-pack split-KV plans. |
| 3 | `flashinfer-ai/flashinfer@4c5d951` (`#7756`) | 2026-10-28 | `minimax_h3_sm120_varlen_attention_fp8` (`[tokens, 56, 128]`) selects overlap width $P$ (tail attention CTAs, $2P$ producer CTAs) from attention-weighted mean segment length $L = \sum s_i^2 / \sum s_i$: $P=64$ when $L \le 1792$ (170 SMs) or $L \le 2304$ (188 SMs); $P=64$ (170 SMs) or $P=48$ (188 SMs) when `padded_tokens > 16384` and $L \ge 4096$; else $P=32$. |
| 4 | `flashinfer-ai/flashinfer@3c7e075` (`#6299`) | 2026-10-28 | Fixes CuTe DSL `GroupedQueryAttentionDecodePaged` page-table ring race by increasing `pt_stages` from `4` to `5` (`assert pt_stages >= prefetch_iters + 3`), eliminating tile `s + 4` V-page aliasing (**30%–50%** relative error, `2e-3` fit to `s+4` oracle) and fixing `reduction="none"` sink rescaling. |
| 5 | `flashinfer-ai/flashinfer@3957698` (`#7695`) | 2026-10-28 | Ships `bs_ue8m0_swap_rp32_s11` and `bs_ue8m0_swap_c1` on `sm_100a` so all **30 programs** (28 host routes: 14 FP32 scale + 14 UE8M0 block scale) of `cake_grouped_fp8_gemm` ship on both `sm_100a` and `sm_103a`, with retuned odd-128 `N` and small-block swap-AB routing. |
| 6 | `flashinfer-ai/flashinfer@ae13e9c` (`#7682`) | 2026-10-28 | Updates Hopper (`sm_90a`) `cake_vsa_sm90` compact GQA programs (`h16-m8192-n8192-k5`, $g \in \{4, 8\}$) with two-step `shfl.sync.bfly` quad exchange, cutting `SHFL` count from `97 -> 65` (BF16, **-32.99%**) and `90 -> 58` (Sage-FP8, **-35.56%**). |
| 7 | `flashinfer-ai/flashinfer@499012e` (`#7694`) | 2026-10-28 | Expands `cake_muse_spark_norm` (hidden `8192`) from `68 -> 77` configs (adding GB300 quarter-row next-RMSNorm for $M \in (1536, 3072]$ and R200 $M > 4096$ row-ring splits) and `cake_tp_moe_finalize_rmsnorm` (hidden `4096`, top-k `8`) from `20 -> 22` configs. |
| 8 | `flashinfer-ai/flashinfer@866313f` (`#6797`) | 2026-10-28 | Adds `flashinfer.comm.allreduce_mhc_post_prenorm`, fusing TP8 MNNVL one-shot all-reduce with 4-stream mHC post-map and 24-logit pre-norm partials across **89 sublayer boundaries** in GLM-5.3-Flash decode (`267 -> 178` kernel launches/step, saving **89 launches/step**). |
| 9 | `triton-lang/triton@d3229a1` (`#12518`) | 2026-10-27 | Reverts premature `lookupNumWarps(op) == 1 -> i32_val(0)` folding in `ConvertWarpSpecializeToLLVM.cpp`, preserving `ttg.warp_id` for `NVGPUToLLVM`. |
| 10 | `triton-lang/triton@7e0b1a0` (`#12492`) | 2026-10-26 | Accelerates AMD software `fp8e4nv -> fp16` upcast on RDNA3/CDNA2/3 by packing pairs into `i32` and multiplying by $2^{15-7} = 256.0$ in FP16 to rebias exponents and normalize subnormals in one step (**up to 3.0x** GEMM speedup on RDNA3). |

---

## 3. Analytical Formulations & Deterministic Verification

### 3.1 CuTe DSL Page-Table Ring Live-Slot Lower Bound

For TMA K prefetch distance $D_{\text{pf}} = \text{prefetch\_iters}$, the TMA K warp writes slot $(s + 1) \bmod N_{\text{pt}}$ at the start of iteration $s$ before waiting on the TMA V barrier, while the TMA V warp may still read slots $s - D_{\text{pf}} - 1 \dots s$:

$$N_{\text{pt}} \ge D_{\text{pf}} + 3 \implies N_{\text{pt}} \ge 5 \quad (\text{for } D_{\text{pf}} = 2)$$

All 5 deterministic invariants in `experiments/flashinfer_mhc_allreduce_fmha_ring_gqa_pt_triton_fp8_phase130/run.py --check` pass (`sm103a_w8_all_rows_faster`, `gb300_4stage_fp8_faster_than_3stage`, `cute_dsl_5stage_pt_ring_eliminates_race`, `vsa_sm90_shfl_reduced_by_32_instructions`, `glm53_mhc_saves_89_launches_per_decode_step`).

---

## 4. Architect Guidance & Falsifiable Follow-Up

- **Blackwell MNNVL mHC Decode**: Always fuse the row-parallel MNNVL one-shot all-reduce with the mHC stream mixing and pre-norm partials (`allreduce_mhc_post_prenorm`) when serving multi-stream residual models like GLM-5.3-Flash at TP8.
- **Falsifiable Follow-Up Question**: At what token batch size does GLM-5.3-Flash TP8 MNNVL transition from latency-bound one-shot all-reduce (`allreduce_mhc_post_prenorm`) to bandwidth-bound two-shot reduce-scatter + all-gather?
