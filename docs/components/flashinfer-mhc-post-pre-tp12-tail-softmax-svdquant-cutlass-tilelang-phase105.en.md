# FlashInfer SM103 `mhc_fused_post_pre` Boundary Fusion, Cooperative `kimi_k3_tp12_tail` Launch, Blackwell 10/12-CTA Cached-Cluster Softmax, SM103a `K=96` NVFP4 SVDQuant MMA, and TileLang Negative-Index Guard (Phase 105)

> **Language / 语言**: [中文 (Chinese)](flashinfer-mhc-post-pre-tp12-tail-softmax-svdquant-cutlass-tilelang-phase105.md) | **English (英文版)**

## 1. Architectural Context & Core Bottlenecks

Across `GLM-5.3-Flash` (45 layers with 4 manifold-constrained hyper-connection residual streams `[T, 4, 4096]` BF16 and **89 `post -> pre` boundaries** per forward pass), `Kimi-K3` TP12 tail reductions, and Blackwell large-vocabulary sampling and `SVDQuant` GEMMs, four kernel-level bottlenecks emerge:
1. **Three-Pass HBM Round-Trips at mHC Boundaries**: The unfused pipeline launches `mhc_post`, a TF32 pre-norm projection GEMM (`[24, 16384]`), and `mhc_pre_big_fuse_with_prenorm` (20 Sinkhorn iterations + RMSNorm), writing `residual_cur` once and re-reading it twice (**`1.69 GB` HBM traffic at `T = 11,476`**).
2. **Infinite Spin-Wait Deadlocks on Undersized Green-Context Partitions**: In `kimi_k3_tp12_tail` (`M <= 8`), single-launch `k123` CTAs spin-wait on peer CTAs of the same grid. Standard launches deadlock indefinitely if a green-context partition cannot hold the entire grid concurrently.
3. **Cluster Size Gaps in Cached-Cluster Softmax & Fixed `K=64` NVFP4 SVDQuant Stages**: Blackwell cached-cluster softmax lacked one-wave `10-CTA` and `12-CTA` programs, while SM103a `cake_nvfp4_svdquant` issued four `K=64` MMAs per 256-wide K stage instead of `96 + 96 + 64`.
4. **CuTeDSL Version Gate Regression & TileLang Out-of-Range Negative Indices**: CUTLASS `blockwise_gemm.py` enabled `opt_level=2` only when the version string equaled `"4.6"`, and TileLang's `LegalizeNegativeIndex` added `extent` once without checking `index + extent < 0`.

## 2. Primary Source Implementation & Mathematical Invariants

- **FlashInfer SM103 `mhc_fused_post_pre`** (`f56fecc7ebf7989a5f075031ff0ded26000b1e8b`): Fuses all three boundary stages into one SM103 kernel for $T \in [33, 16384]$, cutting launches across 89 boundaries from **`267 -> 89` (`-178` launches/pass)**, reducing `T=11,476` HBM traffic from **`1.69 -> 0.94 GB` (`-44.38%`, `1.7979x`)**, and speeding up GB300 decode (`T=48`) by **`2.1628x` (`14.88 -> 6.88 us`)** and mixed prefill (`T=11,476`) by **`1.6694x` (`310.96 -> 186.27 us`)**.
- **Cooperative `kimi_k3_tp12_tail` (`9a27746f1d`), 10/12-CTA Softmax (`e61cdfa925`) & SM103a `K=96` SVDQuant (`4f3b2d6b91`, `c76ffef227`)**: Enforces whole-grid residency via cooperative launch (**`1 -> 0` deadlocks on undersized partitions**); expands cached-cluster softmax from `14 -> 20` programs (**`1.1692x` B200 speedup at `37x128k`**, **`1.1471x` GB300 speedup at `84x100k`**); and reduces SM103a SVDQuant 256-K stage MMAs from **`4 -> 3` (`96+96+64`)**, achieving **`1.0955x` (`162.46 -> 148.30 us`)** on GB300 while guarding `INT64_MIN / -1` in `CDiv`.
- **CUTLASS (`147295a3d4`, `59e3a3338d`, `b87a77760f`) & TileLang (`004b7a2a69`, `fa456b8b2c`)**: Enables `opt_level=2` for CUTLASS `>= 4.7`, gates `SM121` on CUDA 12.9, preserves deterministic link order, flushes L2 on `torch.cuda.current_device()`, and rejects out-of-range negative constant indices (`index + extent < 0`, catching **`3` OOB indices**).

## 3. Deterministic Verification Summary

Run `python experiments/flashinfer_mhc_post_pre_tp12_tail_softmax_svdquant_phase105/run.py --check`:

| Metric | Baseline | Phase 105 | Improvement | Evidence Class |
|---|---:|---:|---:|---|
| GLM-5.3-Flash 89-Boundary Launches | `267` | `89` | **`178` launches saved (`-66.67%`)** | `implementation` + `own-offline` |
| `mhc_fused_post_pre` Traffic (`T=11,476`) | `1.69 GB` | `0.94 GB` | **`-44.38%` (`1.7979x`)** | `author-benchmark` + `own-offline` |
| GB300 Decode (`T=48`) / Mixed (`T=11,476`) | `14.88 us` / `310.96 us` | `6.88 us` / `186.27 us` | **`2.1628x` / `1.6694x`** | `author-benchmark` |
| Cached-Cluster Softmax (`37x128k` B200 / `84x100k` GB300) | `9.95 us` / `14.43 us` | `8.51 us` / `12.58 us` | **`1.1692x` / `1.1471x`** | `author-benchmark` |
| SM103a NVFP4 SVDQuant `(16384,3072,12288,32,1)` | `162.46 us` | `148.30 us` | **`1.0955x` (`4 -> 3` MMAs)** | `author-benchmark` |
