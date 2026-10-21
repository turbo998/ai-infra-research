# FlashInfer Cake GLM-5.3 Four-Stream mHC Training, Blackwell MSA Persistent K-Major Prefill, FP4 AllReduce In-Flight Scale & TRTLLM-Gen Multi-Wave KV Split / BMM Ordering

> **Phase**: Phase 45 (`priority: 54`) · **Primary Sources**: `flashinfer-ai/flashinfer@130dd2429a` (`#6897`), `2ff6d8a5f2` (`#6873`), `3b65505f40` (`#6904`), `c7d1eacfdc` (`#6875`), `a3f2525e39` (`#6331`), `c1bcf70660` (`#6845`)  
> **Deterministic Verification**: `experiments/flashinfer_cake_mhc_msa_kmajor_multiwave_bmm/run.py --check`

---

## 1. Key Architectural Findings & Quantitative Summary

1. **Cake GLM-5.3 Four-Stream mHC Training (`#6897`, `130dd2429a`)**: `flashinfer.cake_mhc_train` implements four-stream multi-head cross-stream connection (`streams=4, hidden=4096`, flattened `u: [T, 16384]`) across `sm_100a`/`sm_103a`/`sm_107a`:
   - **Compact Saved Stats & Zero-Stat Checkpoint Replay**: `forward_prepare` saves `256 B/token` (`64` FP32 scalars), while `checkpoint` mode saves `0 B/token` and reconstructs the per-token statistics inside backward via `forward_stats` before executing `backward_reduce` and `backward_dx`.
   - **Measured Speedups vs `torch.compile`**: On GB300 (`sm_103a`), delivers **`9.31x–15.95x` forward** (`1454.7 us` vs `23197.5 us` at `T=65536`), **`6.61x–8.30x` backward**, and **`9.95x–14.79x` checkpoint** speedups (`2823.8 us` vs `41773.9 us`). On R200 (`sm_107a`), delivers **`7.68x–13.52x` forward**, **`7.51x–8.60x` backward**, and **`10.48x–14.28x` checkpoint** speedups.
2. **Cake Blackwell MSA Persistent K-Major Prefill (`#6873`, `2ff6d8a5f2`) & Kimi-K3 Round-2 AttnRes (`#6904`, `3b65505f40`)**:
   - Adds 3 persistent K-major prefill routes (flat GQA-16, paged GQA-16, paged GQA-8) with exact BF16 combine across `37` route stages per architecture, reaching **`3.39x` geomean speedup** on GB300 (`106` shapes, up to **`5.284x`** on paged prefill and **`3.229x`** on FP8-KV flat decode) and **`3.35x` geomean** on B200 (`105` shapes).
   - Round-2 Kimi-K3 AttnRes repacks `133` kernel programs into `161` multi-symbol libraries (`75` `sm_100a` + `86` `sm_103a`), achieving **`1.0072x` geomean** across `582` GB300 rows (up to **`1.0888x`** at `serving_m5_k5` PDL-off).
3. **Cake FP4 AllReduce In-Flight Row Scale & TP4 SMEM Pass-2 (`#6875`, `c7d1eacfdc`)**:
   - TP2 fused AllReduce + RMSNorm overlaps `ld.volatile.global.u32` row-scale polling (`0x80000000u` sentinel) with the first data sweep via `__ballot_sync`; TP4 two-shot retains pass-1 validated NVFP4 value + FP8 block-scale packets in shared memory, cutting pass-2 Lamport L2 re-reads from **`3,472 B/row` to `16 B/row` (`-99.54%`)**.
4. **TRTLLM-Gen Opt-In 4-Wave KV Split Decode (`#6331`, `a3f2525e39`) & Manifest-Independent BMM Ordering (`#6845`, `c1bcf70660`)**:
   - `FLASHINFER_TRTLLM_GEN_MULTI_WAVE_KV_SPLIT=1` scales KV split decode up to `kMaxNumWavesForKvSplit = 4` waves when 1 wave yields $\le 2$ splits and `base_ctas <= 2 * SMs`, sizing `partialStats`/`partialMax` by `std::max(mMultiProcessorCount, totalNumCtas) * mStepQ` (`9,216` vs `2,368` entries at `B=48, H_kv=2` on B200).
   - Fixes `getDefaultValidConfigIndex` to enforce a strict weak ordering independent of cubin manifest order by comparing `mNumSlicesForSplitK` before a symmetric unsplit/unclustered `Persistent` CTA check, recovering **`1.90x`** on NVFP4 FC2 (`0.135 -> 0.071 ms`) and **`1.14x`** on MXFP4×MXFP8 FC1 (`0.718 -> 0.631 ms`).

---

## 2. Benchmark Summary Table

| Subsystem / Kernel | Hardware / Shape | Baseline | Optimized | Gain |
|---|---|---:|---:|---:|
| Cake mHC Forward (`#6897`) | GB300 (`sm_103a`), `T=65536` | `23,197.5 us` | `1,454.7 us` | **`15.95x`** |
| Cake mHC Checkpoint (`#6897`) | GB300 (`sm_103a`), `T=65536` | `41,773.9 us` | `2,823.8 us` | **`14.79x`** |
| Cake mHC Checkpoint (`#6897`) | R200 (`sm_107a`), `T=65536` | `28,443.5 us` | `1,992.6 us` | **`14.28x`** |
| Cake Blackwell MSA (`#6873`) | GB300 (`106` shapes) / B200 (`105` shapes) | Baseline Cake MSA | Persistent K-Major + FP8-KV | **`3.39x` / `3.35x` geomean** |
| Cake FP4 AllReduce TP4 (`#6875`) | `sm_100a`/`103a`/`107a`, `H=6144` | `3,472 B/row` L2 re-read | `16 B/row` L2 re-read | **`-99.54%` L2 re-read** |
| TRTLLM-Gen BMM NVFP4 FC2 (`#6845`) | B200, `B=8, M=256, N=4096, K=512` | `0.135 ms` (`splitK3`) | `0.071 ms` (`unsplit`) | **`1.90x`** |
| TRTLLM-Gen BMM MXFP4×MXFP8 (`#6845`) | B200, `B=128, M=128, N=2048, K=7168` | `0.718 ms` | `0.631 ms` | **`1.14x`** |
