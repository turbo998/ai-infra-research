# FlashInfer Cake Grouped FP8 GEMM Swap-AB UE8M0, NVFP4 Sparse MLA Decode tcgen05, MXFP8 MegaMoE EP16 Pushed Mailboxes & MiniMax-H3 SM100/SM103/SM120 Audit

> Chinese version: [flashinfer-cake-fp8-gemm-swap-ab-nvfp4-mla-tcgen05-mxfp8-ep16-push-minimax-h3.md](flashinfer-cake-fp8-gemm-swap-ab-nvfp4-mla-tcgen05-mxfp8-ep16-push-minimax-h3.md)
> Evidence grade: `implementation` + `author-benchmark` (upstream source/PR benchmarks) + `own-offline` (deterministic Python 3.12 stdlib simulator, not direct GPU hardware re-execution)

## 1. Core Architectural Questions & Code Evolution

Phase 84 (`priority: 93`) audits 6 first-hand upstream commits in `flashinfer-ai/flashinfer` across `sm_100a` (B200/GB200), `sm_103a` (GB300), and `sm_120a` (RTX 5090 / RTX PRO 6000 Blackwell Server Edition):

1. **`cake_grouped_fp8_gemm` Swap-AB Block-Scaled UE8M0 Programs ([#7342](https://github.com/flashinfer-ai/flashinfer/commit/84f3bb30b29e2a927bfdacdef2bbe7eca5830fc0))**:
   - Adds `swap`, `swap_run32` (for 32-row group alignment), and `swap_carry` programs for power-of-two packed `UE8M0` block scales on `sm_100a` and `sm_103a`.
   - On sparse small-token routing (`M=16000, N=512, K=4096, groups=512, align=32, 2x10 uniform`), `swap_run32` reduces B200 latency from `42.30 us` to `24.58 us` (`new/shipped = 0.581`, **1.721x** speedup) and GB300 slow from `40.56 us` to `23.49 us` (`0.579`, **1.727x** speedup).
2. **`nvfp4_sparse_mla_decode(..., backend="cake")` 35 `tcgen05` Programs ([#7335](https://github.com/flashinfer-ai/flashinfer/commit/a8ec053888bdb597e9a0a3ca9ae57dfe1fc8b900))**:
   - Replaces 7 `mma.sync` programs with **35 `tcgen05` programs** across 7 cluster sizes (`1, 2, 3, 4, 5, 6, 8` CTAs/token) and 5 split bounds (`64, 128, 192, 256, any` keys/CTA) for 352-byte `nvfp4_ds_mla` KV rows.
   - Keeps key tiles resident in Blackwell **Tensor Memory (TMEM)** instead of shared memory, achieving **1.131x** geomean speedup on B200 (up to **1.305x**) and **1.152x** geomean speedup on GB300 (up to **1.375x** at `W=128, topk=2048`: `61.60 us -> 44.80 us`).
3. **`CakeMxfp8MegaMoeEp16` Pushed Mailboxes & Async-Proxy Fences ([#7350](https://github.com/flashinfer-ai/flashinfer/commit/38b6198055ca44baa10600e62b9c513b59ead0c8))**:
   - Converts 16-rank expert-parallel dispatch (`hidden=3072, inter=5120, 512 experts, top-8`) from post-barrier NVLink pulls to pre-barrier **pushed per-source mailboxes** (`[16, T, 3072]` BF16 rows = `6.0 MiB` at `T=64`, plus `[16, T, 8]` routes), direct per-CTA rank-barrier waits, and `fence.proxy.async.global` before TMA loads (`1.014x–1.019x` faster on 16x GB300 NVL72).
4. **MiniMax-H3 SM100/SM103/SM120 Out-Projection & Varlen Attention ([#7314](https://github.com/flashinfer-ai/flashinfer/commit/b4125442a53382fad96d08bbf3eda580c9d58a02), [#7333](https://github.com/flashinfer-ai/flashinfer/commit/53ecbde224f45741a2aa4a0b928a330a502ab15e), [#7334](https://github.com/flashinfer-ai/flashinfer/commit/b8ed186df31b1ad175d6b0b4a47120542c48279d))**:
   - SM100/SM103 NVFP4 out-projection runs partial last-wave tail tiles as `256x128` half-width units (**1.042x** geomean on GB200, up to **1.052x**); SM120 FP8 out-projection walks one M tile per swath (**1.009x** geomean); SM120 FP8 varlen attention adds early PV hand-off and next-unit prefetch (**1.014x–1.055x** on RTX 5090, **1.030x–1.041x** on RTX PRO 6000).

## 2. Reproducible Experiment

- Command: `python experiments/flashinfer_cake_fp8_gemm_swap_ab_nvfp4_mla_tcgen05_mxfp8_ep16_push_minimax_h3/run.py`
- Evidence ID: `ev-flashinfer-cake-fp8-gemm-swap-ab-nvfp4-mla-tcgen05-mxfp8-ep16-push-minimax-h3-2026`
