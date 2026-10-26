# Phase 93 Track 1: FlashInfer Deferred Sampling Reduction, Small-Batch Activation Split, Cake QSA/Rubin MLA & TileLang Top-K Global Workspace

> **中文版 / Chinese Version**: [`flashinfer-sampling-act-qsa-rubin-mla-tilelang-phase93.md`](flashinfer-sampling-act-qsa-rubin-mla-tilelang-phase93.md)  
> **Reproducible Experiment**: [`experiments/flashinfer_sampling_act_qsa_rubin_mla_tilelang_phase93/`](../../experiments/flashinfer_sampling_act_qsa_rubin_mla_tilelang_phase93/)  
> **Upstream Commits Covered (`2026-10-22` – `2026-10-26`)**:
> - `flashinfer-ai/flashinfer@1257bb40fd` (`#5352`): `perf(sampling): defer last-valid-index reduction until fallback`
> - `flashinfer-ai/flashinfer@84fb433aae` (`#5614`): `perf(activation): split act_and_mul rows across blocks for small batches`
> - `flashinfer-ai/flashinfer@055b0e2b5d` (`#7394`): `perf(cake_qsa_nvfp4): the tcgen05 decode's softmax tile maximum through redux.sync.max on compute capability 10.0 / 10.3`
> - `flashinfer-ai/flashinfer@bc8ec9b1ec` (`#7403`): `perf(cake_rubin_mla_nvfp4_decode): a shaved one-wave plan for long full-tile rows on Rubin (sm_107a)`
> - `flashinfer-ai/flashinfer@f526f18477` (`#7401`): `perf(cake_sampling): rounds 12-13 -- regenerated stage-1 radix top-k kernels, no-tail stage-1 twins, one host call per sampling chain, per-device stage-1 picks`
> - `flashinfer-ai/flashinfer@4310f2af15` (`#7395`): `perf(cake_kimi_k3_latent_moe): packed one-partial norm for TP8 prefill tails, 256-byte L2 promotion of the decode weight maps`
> - `flashinfer-ai/flashinfer@31e839e2ac` (`#7383`): `perf(cake_minimax_h3): persistent NVFP4 norm and norm-overlapped NVFP4 FC1 in the MLP block on SM100 / SM103`
> - `flashinfer-ai/flashinfer@225b3b3b8e` (`#6823`): `feat(mla): add SM90 BF16 DOTS3 sliding-window decode`
> - `tile-ai/tilelang@fe8c34618f` (`#3612`): `[Transform] Clone analyzer before vectorization rewriting`
> - `tile-ai/tilelang@4da574c013` (`#3280`): `[Bugfix] Fix candidate buffer overflow in deepseek_v32 topk selector`

---

## 1. Architectural Overview

In low-latency LLM decode and long-context sparse MLA inference, kernel tail latency is frequently dominated by three micro-architectural bottlenecks: **(1) unconditional block-wide reductions on every tile for rare floating-point fallback paths**, **(2) single-CTA-per-row launches at small decode batch sizes that leave most SMs idle and serialize dependent memory round-trips**, and **(3) heterogeneous cluster occupancy tails on next-generation architectures such as NVIDIA Rubin (`sm_107a`, 208 SMs)**. This document analyzes 10 upstream commits across FlashInfer and TileLang.

---

## 2. Deferred Fallback Reduction in Probability Sampling & Cake Sampling Rounds 12–13 (`#5352`, `#7401`)

### 2.1 Deferring `last_valid_index` Block Reduction (`1257bb40fd`)
In `include/flashinfer/sampling.cuh`, `DeviceSamplingFromProb` previously performed an integer `BlockReduce::Max` + `__syncthreads()` on every scanned vocabulary tile (`BLOCK_THREADS = 1024`, `VEC_SIZE = 4` $\Rightarrow 4,096$ tokens/tile) to track the last valid token index in case floating-point rounding prevented the CDF scan from selecting a token.
- For a vocabulary of $V = 151,936$, every row executed $\lceil 151936 / 4096 \rceil = 38$ block-wide reductions even though the fallback is only consumed when total probability mass rounds below the Philox uniform sample $u$.
- By retaining each thread's candidate index in registers across all tiles and executing `BlockReduce::Max` only once after the loop if no token was selected, normal-path block reductions drop from **38 to 0 per row**.

### 2.2 Cake Sampling Rounds 12–13 (`f526f18477`)
- Expands `cake_sampling` to **104 stage-1 radix top-k builds** (adding 35 `no_tail` stage-1 twins across 1/2/4/8-CTA clusters and 4–48 entries/thread) and 15 stage-2/3 sparse top-p builds, issuing a single host call per sampling chain with per-device stage-1 selection across GB300 (`sm_103a`), B200 (`sm_100a`), R200 (`sm_107a`), and H100 (`sm_90a`).

---

## 3. Small-Batch Row Splitting in Activation Kernels (`#5614`, `84fb433aae`)

For `silu_and_mul`, `gelu_and_mul`, and `gelu_tanh_and_mul`, the legacy `act_and_mul_kernel` launched one 256-thread block per token row:
- At `num_tokens = 16` and `d = 14,336` in BF16 (`1,792` 16B vectors/row), only **16 CTAs** ran on a 132-SM GH200 (**12.12%** SM utilization), and each thread executed **7 dependent memory round-trips** (`4.96 us`, `289 GB/s`).
- `act_and_mul_split_kernel` splits each row across `split` consecutive CTAs (`blockIdx.x = row * split + chunk`) when `num_tokens < 2 * num_sms`, sizing `split` so each thread handles at most `kUnroll = 2` vectors (`split = 4`, launching **64 CTAs** for 16 tokens) and issuing both 16B loads per half before conversion and computation—boosting effective bandwidth by **3.15x** (`910.48 GB/s`).

---

## 4. Blackwell/Rubin Kernel Optimizations & SM90 DOTS3 Sliding-Window MLA (`#7394`, `#7403`, `#7395`, `#7383`, `#6823`)

- **Cake QSA NVFP4 `redux.sync.max.f32` (`#7394`, `055b0e2b5d`)**: Replaces the 16-instruction butterfly shuffle-xor max tree in `decode_tcgen05`, `decode_tcgen05_coop`, and `decode_tcgen05_persistent` with hardware `redux.sync.max.f32` on SM100/SM103, achieving up to **1.04x** eager cold-L2 speedup on B200 and GB300.
- **Rubin (`sm_107a`) Shaved One-Wave MLA Plan (`#7403`, `bc8ec9b1ec`)**: An R200 board (`208` SMs) hosts `50` clusters of 4 CTAs (`200` SMs) plus `4` pair clusters (`8` SMs). A single wave of 48 long full-tile rows ($\ge 1,024$ key tiles of 128 keys) previously left **16 SMs idle**. The host planner now prices a **shaved one-wave plan** that carves a short range of `MLA2_SHAVE_TILES = 48` tiles off each full tile to feed the remaining 4-CTA and 2-CTA pair clusters, merging 2 FP32 partials via `merge_w1` and reducing modelled makespan by **6.50%** (`1031 -> 964` tile-times, beating the `3.0%` slack threshold).
- **Kimi-K3 Stable LatentMoE (`#7395`, `4310f2af15`) & MiniMax-H3 Persistent NVFP4 Norm (`#7383`, `31e839e2ac`)**: Adds byte-identical `bf16x2` packed one-partial RMSNorm for TP8 prefill tails ($T \ge 2048$) and 256B L2 TensorMap promotion in Kimi-K3; converts MiniMax-H3's NVFP4 norm + AdaLN + quantization into a persistent row-streaming kernel that PDL-overlaps the NVFP4 FC1 GEMM prologue (**1.053x** faster on B200).
- **SM90 BF16 DOTS3 Sliding-Window MLA (`#6823`, `225b3b3b8e`)**: Consumes DOTS3's Page64 BF16 latent cache directly (`d_qk = 1088`, `d_v = 1024`, window $W = 513$) on Hopper SM90 without expanding K/V across 64 query heads (**123.3x** lower KV read traffic per token).

---

## 5. TileLang Analyzer Cloning & DeepSeek-V3.2 Top-K Buffer Overflow Fix (`#3612`, `#3280`)

- **Clone Analyzer Before Vectorization (`fe8c34618f`, `#3612`)**: Clones `arith::Analyzer` inside `VectorizeLoop` before invoking `VectorizeRewriter`, preventing loop-local bounds from leaking across subsequent IR passes.
- **Dynamic Global Workspace for DeepSeek-V3.2 Top-K (`4da574c013`, `#3280`)**: Replaces the fixed `[2, 4096]` shared-memory candidate staging buffer in `tl_topk_impl` with a caller-allocated `(batch, 2, seq_len)` global-memory workspace, eliminating `CUDA_ERROR_ILLEGAL_ADDRESS` crashes when $>4,096$ tied or common-low-mantissa scores fall into the threshold radix bucket (up to `32,768` candidates at `seq_len = 32,768`).
