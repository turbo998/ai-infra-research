# Deep Dive: FlashInfer `cake_causal_conv1d` Dynamic Atomic Counter, Kimi-K3 AttnRes SM103 384-Thread Pair, Grouped/Batch FP8 GEMM & CUTLASS SM110a / TileLang Bounds (Phase 104 · Priority 113)

> **中文版本**: [flashinfer-causal-conv1d-kimi-attnres-grouped-fp8-cutlass-tilelang-phase104.md](flashinfer-causal-conv1d-kimi-attnres-grouped-fp8-cutlass-tilelang-phase104.md)
> **Experiment**: [`experiments/flashinfer_causal_conv1d_kimi_attnres_grouped_fp8_phase104/run.py`](../../experiments/flashinfer_causal_conv1d_kimi_attnres_grouped_fp8_phase104/run.py) · **Summary**: [`summary.json`](../../experiments/flashinfer_causal_conv1d_kimi_attnres_grouped_fp8_phase104/summary.json)
> **7-Card Visual Report**: [`docs/reports/ai-infra-phase104-benchmark-cards.html`](../reports/ai-infra-phase104-benchmark-cards.html)

---

## 1. Evidence Classification & Upstream Anchors

| Source ID | Repo & Commit | Path | Classification |
|---|---|---|---|
| `src-flashinfer-cake-causal-conv1d-grad-speedup-2026` | `flashinfer-ai/flashinfer@c289936e6b8fd07d751dae48db2a976bb58f9cc8` | `flashinfer/cake_causal_conv1d_train.py` | `implementation` + `author-benchmark` |
| `src-flashinfer-kimi-k3-attn-res-sm103-pair384-2026` | `flashinfer-ai/flashinfer@53c4cd234118c053336d77ff7df572469be77fbe` | `flashinfer/experimental/cake_kimi_k3_attn_res/cake_backend.py` | `implementation` + `author-benchmark` |
| `src-flashinfer-grouped-fp8-ue8m0-swap-rp32-2026` | `flashinfer-ai/flashinfer@216d52e164c881f591481aa3108e5d245168667c` | `csrc/cake_grouped_fp8_gemm/..._kernel.cu` | `implementation` + `author-benchmark` |
| `src-flashinfer-batch-deepgemm-fp8-masked-2cta-2026` | `flashinfer-ai/flashinfer@7e2ee7610470492fa2fe05da9e16661e92e4a3c6` | `csrc/cake_batch_deepgemm_fp8/..._kernel.cu` | `implementation` + `author-benchmark` |
| `src-flashinfer-blackwell-msa-shared-tu-deslop-2026` | `flashinfer-ai/flashinfer@4a439a7e645f2d913635254b283328675d29a8f4` | `csrc/blackwell_msa/..._kernel.cu` | `implementation` |
| `src-cutlass-sm110a-cta-reconfig-activated-2026` | `NVIDIA/cutlass@78e46228a1bfc5c972e2f9a634223af784cd6949` | `include/cutlass/arch/reg_reconfig.h` | `implementation` |
| `src-cutlass-tmem-allocator-constants-compat-2026` | `NVIDIA/cutlass@f614dc40e17fb3ddb1b7b474318b48a8d5d21a2c` | `include/cute/arch/tmem_allocator_sm100.hpp` | `implementation` |
| `src-tilelang-sliced-fill-end-bound-fix-2026` | `tile-ai/tilelang@323b02ac2a9d309dae7a00939da4b7f6463a5ba9` | `src/op/fill.cc` | `implementation` |
| `src-tilelang-ws-neg1-all-threads-fix-2026` | `tile-ai/tilelang@ee4151896de44e29c109e543b0e04c109e543b0e` | `src/ir.cc`, `tilelang/cuda/language/warpgroup.py` | `implementation` |
| `src-tilelang-ascend-per-core-conditionals-2026` | `tile-ai/tilelang@060e1ff31bbb0b65f27286fafc4b1953e109c2b1` | `src/ascend/transform/materialize_schedule_units.cc` | `implementation` |

---

## 2. Implementation Paths, Data Layout & Synchronization

### 2.1 `cake_causal_conv1d_train` Dynamic Atomic Chunk Counter & Zero-Left Workspace (`c289936e6b`)
- **Bottleneck**: Under packed variable-length training batches (e.g., `[4096, 2048, 1024, 512]` tokens), static equal-share partitioning across CTAs in a channel tile stalls the wave on the `4096`-token document CTA.
- **Mechanism**: CTAs claim `256`-token chunks dynamically via a device `int32` atomic counter and execute paired FP32 math, cutting max CTA tokens from `4096` to `2048` and achieving a **`2.2505x` (`4096 -> 1820` cycles)** deterministic speedup (`1.052x–1.926x` author-reported across all `528` test rows on B200, `1.053x–1.762x` on GB300, `1.037x–1.741x` on R200). The kernel resets its counter to zero upon completion so a pre-zeroed `int32` workspace tensor can be reused across CUDA Graph replays with **`0` extra `cudaMemsetAsync` launches**.

### 2.2 Kimi-K3 Attention Residuals SM103 384-Thread 3-Warpgroup Pair (`53c4cd2341`)
- Upgrades `M = 192, K = 4, 6` on SM103 (`B300` / `GB300`) from `320` threads (`168`-reg cap, `28` spilled regs/lane at `K = 4`) to `384` threads (`8` consumer warps at `232` regs + `4` producer warps at `40` regs = **`64,512 <= 65,536` regs/CTA**), eliminating `100%` of register spills at `K = 4` and achieving **`1.0575x` (B300)** and **`1.0553x` (GB300)** speedups.

### 2.3 Grouped/Batch FP8 GEMM & MSA Shared TUs (`216d52e164`, `7e2ee76104`, `4a439a7e64`)
- `bs_ue8m0_swap_rp32` 32-row-block swap-AB speeds up $M=1152, N=1024, K=1024$ on B200 from `7.82 -> 6.86 us` (**`1.1399x`**); masked `BLOCK_N` + 2-CTA builds speed up `cake_batch_deepgemm_fp8` from `4.35 -> 3.63 us` (**`1.1983x`**); `cake_blackwell_msa` folds near-duplicate CUDA kernels into `-D`-selected shared translation units.

### 2.4 CUTLASS SM110a `setmaxnreg` & TileLang Bounds (`78e46228a1`, `323b02ac2a`, `ee4151896d`)
- Enables `CUDA_CTA_RECONFIG_ACTIVATED` on `SM110a`; bounds TileLang sliced `T.fill` by `min + extent <= shape` (`4` OOB SMEM elements prevented on `As[6:12]` of `shape=8`); fixes `T.ws(-1)` so all `256` threads execute instead of `0` threads in `[-128, 0)`.
