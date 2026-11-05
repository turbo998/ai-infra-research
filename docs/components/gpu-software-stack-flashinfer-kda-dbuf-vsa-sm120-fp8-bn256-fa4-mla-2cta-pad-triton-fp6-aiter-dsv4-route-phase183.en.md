# GPU Software Stack: FlashInfer CuTe KDA Double-Buffered Barrier, Cake SM120 Block-64 VSA Sparse Attention, SM107a Grouped FP8 `BLOCK_N=256` & 216-SM `K1` Wgrad, FlashAttention-4 2CTA In-Kernel MQA Head Padding, Triton Gluon FP6 Shared Tiles & AITER DeepSeek-V4.1 Fused Route `-1` Pad Drop (Phase 183)

> 🌐 **Language / 语言切换**: **[中文](gpu-software-stack-flashinfer-kda-dbuf-vsa-sm120-fp8-bn256-fa4-mla-2cta-pad-triton-fp6-aiter-dsv4-route-phase183.md)** | **[English](gpu-software-stack-flashinfer-kda-dbuf-vsa-sm120-fp8-bn256-fa4-mla-2cta-pad-triton-fp6-aiter-dsv4-route-phase183.en.md)**

---

## 1. Core Architectural Problems & Primary Source Evidence (Pillar 0 & Pillar 1)

Phase 183 (`priority: 192`) audits **11 upstream mainline commits** across Blackwell (`sm_100a` / B200 / GB200), Blackwell Ultra (`sm_103a` / GB300), RTX PRO 6000 Blackwell (`sm_120`), Rubin (`sm_107a` / R200), and AMD `gfx1250` (MI455X) kernel and compiler stacks:

1. **FlashInfer CuTe DSL KDA Double-Buffered `update_ready` Barrier & Cake SM120 `vsa_sm120_blk64` Persistent Sparse Attention**:
   - `0102aa9f9f46` (#8684) resolves a multi-chunk phase-parity deadlock in the CuTe DSL KDA decomposition chain kernel (`fused_kda_64_chain_k_dbuf_cute`) where a single `update_ready` barrier allowed Warp 12 on chunk $c+1$ to arrive before Warp 13 or the CG warps completed their wait on chunk $c$. Indexing `update_ready` across `2` barrier slots (`slot = chunk % 2`, `parity = (chunk // 2) % 2`) passes **`309 / 309`** parameterized GB300 tests even under an injected `2 ms` Warp 12 delay.
   - `7729459ffe1a` (#8799) switches RTX PRO 6000 Blackwell (`sm_120`, 188 SMs) Video Sparse Attention (`cake_vsa_sm120`) to the persistent `vsa_sm120_blk64` backend (two 128-thread 4-warp CTAs/SM or one 8-warp CTA instead of non-persistent 1-CTA-per-output-block `vsa_attention_128x64`), achieving a **`1.084x`** geometric-mean speedup across 22 shapes, up to **`1.263x`** on `tiny_s1024_k2` (`14.3 -> 11.4 us`) and **`1.229x`** on `minimaxh3_prefix_s109632_h7_p8_k64` (`6,879.9 -> 5,599.8 us`) with `max_abs <= 1.95e-3`.
2. **FlashInfer SM107a Single-Run Grouped FP8 `BLOCK_N=256`, 216-SM `K1` Weight Gradient, Plain FP32 SIMT Router Backward & SM103a `H2048` Det-Pair Cap**:
   - `f0ec379398e0` (#8809) widens the single-run UE8M0 grouped FP8 GEMM `BLOCK_N=256` dispatch on R200 (`sm_107a`, 212/216 SMs) to all shapes with `num_k_blocks < 32` (up from `< 16`), cutting 212-SM R200 latency on `fib_q35b_gate_up_t4096` (`E=16, T=4096, N=1024, K=4096`) from `84.85 us` to **`76.17 us` (`1.114x`)**.
   - `f8bedda6038a` (#8782) restores `K1` weight-gradient routes in `cake_dense_projection_gemm` on 216-SM `sm_107a` parts (`sm_count >= 208`), delivering **`1.06x–2.23x`** wgrad speedups (`81.58 -> 36.65 us` = **`2.23x`** at `4096->512, T=16173`) and **`1.03x–1.71x`** full-backward speedups (`110.22 -> 64.59 us`).
   - `8baa822f846e` (#8763) replaces compensated Dot2/Kahan router backward with plain FP32 FFMA SIMT windows (`simt_n128m64s2` / `simt_m32n64s8`) on 148-SM `sm_100a` (GB200/B200) and `sm_103a` (GB300) for `N <= 512`, achieving **`1.24x–3.05x`** over compensated kernels and **`1.04x–5.37x`** over deterministic Cake while keeping `rel_fro <= 2.1e-7` (well below the `2e-6` budget).
   - `386275f02fc6` (#8810) caps the GB300 (`sm_103a`) `H=2048` deterministic-pair RMSNorm backward window at `T=58,496` (eliminating the `1.0622x` `dw` budget breach at `T=58,560` and reducing the worst ratio to **`0.9454x`**) while outperforming non-Cake SOTA by **`1.1008x–1.3441x`**.
3. **FlashAttention-4 Dense 2CTA In-Kernel MQA Head Padding, Triton Gluon FP6 Shared Tiles, TileLang Host-Mapped Zero-Copy & ROCm AITER DeepSeek-V4.1 Route `-1` Pad Drop**:
   - `2ed4a1f0fc0e` (#3140) pads non-power-of-two dense MQA head counts in-kernel (`96 -> 128`, `48 -> 64`) and routes `<= 32` heads to 1CTA in FlashAttention-4 absorbed MLA, accelerating B200 24-head prefill (`B=4, Sq=Sk=4096`) from `4.38 ms` to **`2.14 ms` (`2.047x`)** and 96-head decode (`B=128, Sk=8192`) from `270.1 us` (1CTA) to **`175.5 us` (`1.539x`)**.
   - `d86c8b2f0064` (#12548) adds FP6 (`e2m3` / `e3m2`) shared-memory tiles in Triton Gluon for Blackwell `tcgen05` and TMA: `b6x16_p32` TMA loads (12B packed data + 4B padding per 16B atom) feed `tcgen05_mma` operand B, while `b6p2x16` TMA stores compact 128 lanes into 96B in global memory (**`25.0%` bandwidth reduction vs FP8**); `51f4f818afc1` (#12524) enables LDS prefetch for `gfx1250` `scaled_dot`; TileLang `fd5305807bfd` (#3802) adds explicit `cuMemHostGetDevicePointer_v2` zero-copy host-mapped buffers.
   - ROCm AITER `d0fd39399b46` (#7365) fixes an out-of-bounds LDS and stage-1 memory fault in FlyDSL's fused TP `moe_route_psum_fused` kernel on `gfx1250` (MI455X) when `topk_ids == -1` by writing `DROPPED_ROUTE_ROW` and skipping negative routes, completing **`1,105 / 1,105`** prompts with **`0` faults** on 4x MI455X running DeepSeek-V4.1-Flash + DSpark at **`420.0 tok/s/GPU`** (`1,680.1 tok/s` total), TTFT p50 **`49.6 ms`**, and TPOT p50 **`2.30 ms`**.

---

## 2. End-to-End Dataflow & Architecture Topology (Pillar 1)

```
+--------------------------------------------------------------------------------------------------------+
| Phase 183 GPU Software Stack: Blackwell GB300/B200, RTX PRO 6000 (sm_120), Rubin R200 & MI455X gfx1250 |
+--------------------------------------------------------------------------------------------------------+
  [FlashInfer CuTe KDA Double-Buffer, Cake SM120 VSA & SM107a/SM103a GEMM/RMSNorm (#8684, #8799, #8809..)]
    |-- CuTe KDA Decomp (#8684)      --> 2-Slot update_ready (chunk%2, parity=(chunk//2)%2, 309/309 GB300)
    |-- Cake SM120 VSA (#8799)       --> Persistent vsa_sm120_blk64 (1.084x Geomean, up to 1.263x / 1.229x)
    |-- Cake SM107a FP8 GEMM (#8809) --> R200 Single-Run UE8M0 BLOCK_N=256 for K_blocks<32 (1.114x @ T=4096)
    |-- Cake 216-SM K1 Wgrad (#8782) --> sm_107a K1 Weight-Gradient Route (up to 2.23x wgrad / 1.71x bwd)
    +-- SIMT Router & Det-Pair (#8763)--> Plain FFMA Router Bwd (3.05x vs Comp) + GB300 H2048 Cap T<=58,496
                                      |
                                      v
  [FlashAttention-4 2CTA MQA Pad + Triton Gluon FP6 + TileLang + ROCm AITER DSV4 (#3140, #12548, #7365)]
    |-- FA4 2CTA MQA Pad (#3140)     --> In-Kernel Pad 96->128 / 48->64 & <=32h 1CTA (2.047x Prefill, 1.539x Dec)
    |-- Triton Gluon FP6 (#12548)    --> b6x16_p32 TMA Load + b6p2x16 TMA Store (96B/128 elems, -25% BW vs FP8)
    |-- TileLang Host-Mapped (#3802) --> cuMemHostGetDevicePointer_v2 Zero-Copy Host-Mapped Buffer Codegen
    +-- AITER DSV4 FlyDSL (#7365)    --> Drop -1 Pad Routes in moe_route_psum_fused (420 tok/s/GPU, 0 Faults)
```

---

## 3. Deterministic Offline Verification (Pillar 2 & Pillar 3)

Verified via [`experiments/flashinfer_kda_dbuf_vsa_sm120_fp8_bn256_fa4_mla_2cta_pad_triton_fp6_aiter_dsv4_route_phase183/run.py`](../../experiments/flashinfer_kda_dbuf_vsa_sm120_fp8_bn256_fa4_mla_2cta_pad_triton_fp6_aiter_dsv4_route_phase183/run.py) `--check`:
- `kda_update_ready_barrier_slots = 2`, `kda_gb300_passed_tests = 309`
- `vsa_sm120_geomean_speedup_22shapes = 1.084`, `vsa_sm120_max_speedup = 1.263`, `vsa_sm120_minimaxh3_speedup = 1.229`
- `cake_grouped_fp8_sm107a_bn256_speedup = 1.114` (`K_blocks < 32`), `cake_sm107a_216sm_k1_wgrad_max_speedup = 2.23`
- `cake_simt_router_vs_comp_max_speedup = 3.05`, `cake_simt_router_vs_det_max_speedup = 5.37`, `cake_rmsnorm_sm103a_h2048_det_pair_cap_t = 58496` (`worst_dw_ratio = 0.9454`)
- `fa_mla_24h_prefill_speedup = 2.047`, `fa_mla_96h_decode_2cta_vs_1cta_speedup = 1.539`
- `triton_gluon_fp6_global_bytes_per_128_elems = 96`, `triton_gluon_fp6_vs_fp8_bw_saving_pct = 25.0`
- `aiter_dsv4_mi455x_tok_per_s_per_gpu = 420.0`, `aiter_dsv4_mi455x_ttft_p50_ms = 49.6`, `aiter_dsv4_mi455x_tpot_p50_ms = 2.30` (`1,105` requests, `0` faults)

---

## 4. Enterprise Business Scenario Playbook (Pillar 4)

| Scenario Dimension | ✅ Good Fit | ⛔ Not For |
| :--- | :--- | :--- |
| **GB300 / R200 Hybrid Attention (KDA + MLA) & MoE Router Training** | Enable 2-slot `update_ready` in CuTe KDA (`#8684`), plain FP32 SIMT router backward on 148-SM B200/GB300 (`#8763`, up to `3.05x`), and `<32` K-block `BLOCK_N=256` grouped FP8 GEMM (`#8809`) + `K1` wgrad (`#8782`, `2.23x`) on 212/216-SM R200. | Extending GB300 `H=2048` deterministic-pair RMSNorm backward past `T=58,496` (`T=58,560` breaches the `dw` budget at `1.0622x`; fall back to atomic accumulation per `#8810`). |
| **Blackwell Absorbed MLA 2CTA Decode & RTX PRO 6000 VSA Sparse Attention** | Use FA4 2CTA in-kernel head padding for `48 / 96` head absorbed MLA (`#3140`, `1.539x` decode) with `<=32` heads routed to 1CTA (`2.047x` prefill), and persistent `vsa_sm120_blk64` on `sm_120` GPUs (`#8799`, `1.229x` long prefix). | Padding `<= 32` dense MQA heads onto a 64-row 2CTA tile (wastes half the cluster MMA rows) or using `tcgen05` `b6x16_p32` FP6 TMA layouts outside `sm_100a`/`sm_103a` (`#12548`). |
| **AMD MI455X (`gfx1250`) DeepSeek-V4.1 Dynamic Expert Routing Serving** | Deploy FlyDSL `moe_route_psum_fused` with `DROPPED_ROUTE_ROW` negative-route filtering on 4x MI455X for DeepSeek-V4.1-Flash + DSpark (`#7365`, `420.0 tok/s/GPU`, TTFT `49.6 ms`, TPOT `2.30 ms`). | Casting unguarded `topk_ids == -1` padding routes directly to unsigned indices into LDS `expert_Token_num` (causes stage-1 out-of-bounds writes and GPU memory faults). |
