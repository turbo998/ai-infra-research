# FlashInfer NVFP4 MLA Short-Unit Decode & Split-KV Combine, Batched DeepGEMM FP8, GDN Fused Training, SM90 VSA Retuning, PrimTS Keeps Softmax & 2-CTA MLA Page-ID Prefetch, SM107 Split-K=8 BF16 GEMM, Kimi K3 Held-Release, and TileLang Bulk TMA / NVRTC TVM-FFI Deep Dive (Phase 89)

> **中文版**: [flashinfer-nvfp4-mla-deepgemm-gdn-vsa-prims-sm107-tilelang-phase89.md](flashinfer-nvfp4-mla-deepgemm-gdn-vsa-prims-sm107-tilelang-phase89.md)  
> **Evidence Class**: `implementation` (pinned commit source review) + `author-benchmark` (upstream PR measurements, not re-run on local hardware) + `own-offline` (`experiments/flashinfer_nvfp4_mla_deepgemm_gdn_vsa_prims_sm107_tilelang_phase89/run.py` deterministic stdlib model)

---

## 1. Architectural Context & Problem Statement

On NVIDIA Blackwell (SM100 B200, SM103 B300/GB300, SM107) and Hopper (SM90 H200), raw Tensor Core MMA throughput has scaled so aggressively that production kernel latency is increasingly governed by **serial TMEM/SMEM dependencies between MMA and softmax, global-memory (GMEM) page-table lookups on the critical path to TMA issue, low SM occupancy on narrow/small-M shapes, and JIT binary cache key collisions**. This document analyzes 10 non-overlapping first-hand commits landed on `2026-10-26` across `flashinfer-ai/flashinfer` and `tile-ai/tilelang`.

---

## 2. Kernel & Compiler Deep Dive

### 2.1 `cake_nvfp4_mla_decode`: Short-Unit Decode & General Split-KV Combine (`flashinfer@5e8921c0d5`, PR #7418)
- **Implementation Path**: `flashinfer/cake_nvfp4_mla_decode/csrc/cake_nvfp4_mla_decode.cu` introduces two structural optimizations for NVFP4 block-scaled MLA decode:
  1. **Short-Unit Decode Path**: Tightens pipeline drain and accumulator initialization when $\text{KV}_{\text{len}} \le 4096$ on SM100 (B200) or $\text{KV}_{\text{len}} \le 8192$ on SM103 (GB300).
  2. **General Split-KV Combine (`split_kv >= 3`)**: Replaces the fallback path beyond 2-way split with a vectorized general combine kernel supporting $\ge 3$ KV splits per row.
- **Author Benchmarks (`author-benchmark`)**:
  - **GB300 (SM103)**: 3-way split drops from `12.53 us` to `11.30 us` (**`1.109x`**); 2-way split from `11.94 us` to `11.20 us` (**`1.066x`**).
  - **B200 (SM100)**: 3-way split drops from `13.04 us` to `11.68 us` (**`1.116x`**); 2-way split from `12.32 us` to `11.71 us` (**`1.052x`**).

### 2.2 `cake_batch_deepgemm_fp8`: Masked Batched FP8 GEMM on SM100/SM103 (`flashinfer@e23951f199`, PR #7425)
- **Implementation Path**: Optimizes persistent tile scheduling and scale-factor TMA prefetching for masked batched FP8 GEMM across 164 shapes, eliminating empty-expert barrier stalls.
- **Author Benchmarks (`author-benchmark`)**: Faster on 51 B200 shapes, 54 GB300 #1 shapes, and 51 GB300 #2 shapes, with **0 regressions**:
  - `b128_m128_n512_k128`: `49.41 -> 13.86 us` on GB300 #2 (`PR/main = 0.2804`, **`3.565x` speedup**) and `34.62 -> 14.11 us` on B200 (**`2.454x` speedup**).
  - `sglang_p10_b32_m256_em1_n4096_k7168_ep8`: `5.18 -> 2.46 us` on B200 (**`2.106x`**) and `4.99 -> 2.40 us` on GB300 (**`2.079x`**).

### 2.3 `cake_gdn`: Gated DeltaNet Forward Prep, Tape Stores & Fused `dq/dk/dg` (`flashinfer@445980cc61`, PR #7423)
- **Implementation Path**:
  - Fuses forward state preparation, state tape stores, and backward `dq/dk/dg` computation in `flashinfer/experimental/cake_gdn_train/` (`dynamic_smem_bytes: 91,136 -> 99,328`).
  - Adds a 64-bit `CheckedMul` overflow guard on `(total_tokens, H, D, K)` strides to prevent 32-bit index wraparound on long-context Context Parallelism (`CP=2..16`) runs.
- **Author Benchmarks (`author-benchmark`)**: Across 60 rows (`(H_qk, HV)=(16, 48), D=128, CP ∈ {2, 4, 8, 16}`), achieves **`1.064x`** forward and **`1.042x`** full training step geomean speedup on GB300, and **`1.067x`** forward / **`1.043x`** training step on GB200.

### 2.4 `cake_vsa_sm90`: Hopper FP8 Grouped-Query VSA Retuning (`flashinfer@32a26fe723`, PR #7412)
- **Implementation Path**: Reduces L2 TMA prefetch distance (`2 -> 1`) and in-flight V-transposer `ldmatrix` chunks (`4 -> 1`) on the compact static program (`51.824 -> 49.824 us`, **`1.0402x`** on `g=8`) and disables L2 prefetch (`1 -> 0`) on the shared-transposer queue program.

### 2.5 `prims-ts`: Breaking the Keeps Softmax Chain & Grouped-Q Paged Selection (`flashinfer@7ccb13b8e7`, PR #6937)
- **Root Cause**: In a two-instance `Q128/KV128` Keeps tile, $P$ previously overwrote $S$ in TMEM, serializing $QK^T \to \text{Softmax} \to PV$ and leaving the Tensor Core idle during softmax.
- **Four Architectural Fixes**:
  1. **16-bit K/V K32 Fragment Handoff**: Softmax hands $P$ to PV MMA in four `K32` fragments rather than waiting for the full row, overlapping PV MMA with softmax and shifting part of `exp2` onto the SM100 FMA pipe;
  2. **8-bit K/V (FP8 & Sage) SMEM P Staging**: Places $P$ in SMEM while holding scores in registers so $S$ in TMEM is freed immediately after the max pass, letting the next $QK^T$ run alongside exponentials;
  3. **Softmax Anchor Headroom (`softmax_rescale_threshold_log2`)**: Retains the row exponent anchor when the row maximum grows by at most `threshold` binades (default 8), skipping most in-place output rescales;
  4. **Planner Extension**: Enables grouped `Keeps Q128` for dense and 16-bit multi-token paged decode (`page_size=32`), avoiding per-query-token KV re-reads in `Swaps` tiles.
- **Author Benchmarks (`author-benchmark`)**:
  - Dense BF16 (`B=2, S=75600, H=40, D=128`): **`1.29x`** on B200 (`336.6 -> 261.9 Mcycles`) and **`1.42x`** on GB300 (`293.6 -> 206.5 Mcycles`).
  - Paged decode plan switch (`dense BF16, Hq/Hkv=8, Sq=256, B=64`): **`5.5x`** on B200 (`26,372 -> 4,786 us`) and **`5.6x`** on GB300 (`25,797 -> 4,626 us`).

### 2.6 `prims-ts`: Page-ID Window Prefetch for FP8 2-CTA Dense MLA (`flashinfer@b6694615b3`, PR #7019)
- **Super-Linear Slowdown Diagnosis**: In `throughput_2cta` FP8 dense MLA decode, K/V loader warps previously read page IDs from GMEM *after* acquiring a free SMEM slot, putting full GMEM latency on the critical path to TMA issue (`49%` QK MMA wait-for-K stall and `19%` K-loader page-table stall in NCU).
- **Register Double-Window Prefetch**: Each FP8 K/V loader warp holds a 32-entry register window (`cached_window_page` and `cached_next_window_page`, 1 entry per lane) covering $32 / \text{pages\_per\_k\_tile}$ K tiles, issuing the next window load at the start of the current window before waiting on SMEM slots.
- **Author Benchmarks (`author-benchmark`, B200 `H128, K8192`)**:
  - `page=16, B=128`: `339.73 -> 141.71 us` (**`2.397x` speedup**);
  - `page=32, B=128`: `198.31 -> 135.78 us` (**`1.461x` speedup**).

### 2.7 SM107 Split-K=8 BF16 GEMM, Kimi K3 Held-Release & TileLang TMA/NVRTC (PR #7064, #7411, #3683, #3285)
- **SM107 Split-K=8 & 18-Stage Pipeline (`flashinfer@39c34abafc`, PR #7064)**: Exploits SM107's larger per-CTA SMEM (`_MAX_AB_STAGES: 12 -> 18`) and adds `split_k = 8` for narrow long-K BF16 GEMMs (`M <= 32`), cutting latency by `5%–8%` (`muse-spark pooler_slice TP4` `M=8`: `4.94 -> 4.61 us`) while rejecting non-portable `split_k ∈ {12, 16}` (`1.3x–11x` slower).
- **Kimi K3 Held-Release on SM103 (`flashinfer@0aec9936bd`, PR #7411)**: Holds buffer release until the end of the chunk for `K=5` dense and `K=1` snapshot-write programs on B300/GB300 (**`1.0236x`** on GB300 `M=1024, K=5`).
- **TileLang Bulk TMA Cache Hints & NVRTC TVM-FFI (`tilelang@8ab967ff46` #3683, `tilelang@55743ac373` #3285)**: Restores `CacheHintSm90` (`EVICT_NORMAL`, `EVICT_FIRST`, `EVICT_LAST`) on descriptorless bulk TMA loads (`cp.async.bulk.shared::cta.global`) and adds NVRTC compilation via TVM-FFI with compiler name + version embedded into `cuda_binary_cache.py` keys.

---

## 3. Cross-Architecture Comparison (Fireworks / Together / Cerebras / Groq)

| Dimension | NVIDIA Blackwell + FlashInfer/TileLang | Fireworks AI / Together AI | Cerebras WSE-3 | Groq LPU |
| :--- | :--- | :--- | :--- | :--- |
| **Multi-Token Paged Decode** | PrimTS `Keeps Q128` + 32-entry register page-table prefetch yields up to `5.6x` / `2.40x` speedup | Relies on grouped-query tiles for speculative tree verification | On-wafer SRAM avoids HBM page tables but is capacity-bounded | Compiler-scheduled SRAM streams avoid dynamic page tables |
| **Masked MoE Batched GEMM** | `cake_batch_deepgemm_fp8` delivers up to `3.56x` speedup with zero regressions across 164 shapes | Critical for multi-tenant MoE/LoRA TPOT SLOs | Dataflow fabric routes sparse tokens across cores | Deterministic VLIW execution cannot dynamically skip empty experts |

---

## 4. Architect Exercises & Falsifiable Questions

1. **Page-Window Geometry Calculation**: With a `128`-token K tile and a `32`-lane warp register window, how many K tiles are covered per window for `page_size=16` vs `page_size=64`? Why does `make_mla_decode_config` enforce `WARP_LANES % pages_per_k_tile == 0`?
2. **Falsifiable Follow-Up**: Recalibrating the 16-bit `Keeps Q128` cost-model mainloop factor (currently `3.2`, tuned prior to the `1.29x–1.42x` K32 fragment handoff speedup) should shift additional `Sq ∈ [4, 16]` multi-token paged decode shapes from `Swaps` to `Keeps Q128`.
