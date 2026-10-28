# FlashInfer KDA Plan Caching, Rubin SM107 SSD Combined, Muse Spark Norm, MM FP4 2x2 Multicast, Grouped FP8 SiLU CTA-Cap, BF16-FP4 PDL Alpha, SM120 Weight-First, GDN Det Train, Softmax Balanced Chunks, mHC Release & TileLang Bounds Safety Deep Dive (Phase 124)

> **中文版**: [flashinfer-kda-plan-ssd-sm107-muse-mm-fp4-silu-bf16fp4-sm120-gdn-softmax-mhc-tilelang-phase124.md](flashinfer-kda-plan-ssd-sm107-muse-mm-fp4-silu-bf16fp4-sm120-gdn-softmax-mhc-tilelang-phase124.md)
> **Evidence Tier**: `implementation` + `author-benchmark` + `own-offline` (Zero GPU hardware re-measurement claims)

---

## 1. Architectural Context & Problem Statement

Across Blackwell (`sm_100a` B200 / GB200, `sm_103a` B300 / GB300, `sm_120a` RTX PRO 6000) and Rubin (`sm_107a`) deployments, Phase 124 Track A addresses ten concrete kernel and compiler bottlenecks:

1. **Repeated `plan()` & Host Synchronization in `cake_kda_train` (`bd60c11809`)**: Calling `chunk_kda` without a caller-owned workspace reran 63 `select_record` calls (`1.044 ms`) on every step and triggered `2–6` host synchronizations under `KdaArena(lean=True, tail_passes=4)` and batched documents, stalling the host for `101.45–105.88 ms` behind a 98 ms GPU spin.
2. **Rubin `sm_107a` Mamba2 SSD Cluster Sizing & Single-Chunk Tail Handoff (`ecd4d71d89`)**: Multi-CTA clusters reduced occupancy on intermediate varlen shapes on `sm_107a`, and `tail_chunks == 1` varlen exact-prefix lacked a direct runtime-ticket flag-word handoff.
3. **GB300 / B200 Gated Residual RMSNorm & Next-RMSNorm Routing Gaps (`d9447704cb`)**: `cake_muse_spark_norm` (`H=8192`) needed PDL-regime retuning on GB300 and 4-CTA 2-row L2-prefetch / L2 load-hint routing on B200 (`M = 1152..4096`).
4. **GB300 FP4 GEMM Scale-Factor Bandwidth at `M = 1793..2048` (`c9fc6cab40`)**: `cake_mm_fp4` on 152-SM GB300 required cluster-of-8 `2x2` multicast (`_sfr_mc42`) scale-factor ring programs.
5. **128-CTA Cap on Grouped FP8 Fused SiLU Quant (`a0384efcba`)**: `FUSED_MIXED_ROUTE` capped launches at 128 CTAs, leaving 20–24 SMs idle on B200 (148 SMs) and GB300 (152 SMs).
6. **Non-Coherent `alpha[0]` Load Hazard Under PDL in `cake_blackwell_bf16_fp4` (`464ca3c813`)**: Small-M (`16x1024x1024`) `blackwell-native` PDL programs used a read-only non-coherent load for `alpha[0]` that the compiler could hoist before `griddepcontrol.wait`.
7. **Serial Weight TMA and Activation Gather on SM120 (`1e7a60d232`)**: `frost_low_latency` grouped FP8 MoE (`H=2048, I=512, top_k=10`) needed `weight_first=True` to overlap weight TMA loads with software activation gather.
8. **Deterministic GDN Host Dispatch, Softmax Whole-Tile Imbalance & mHC `dY` Retention (`4cc027c249`, `f4be6e9df5`, `1fb63709b8`)**: `cake_gdn_det_train` lacked longest-first sequence dispatch and 2-stream tail fill; `cake_softmax` assigned whole 2048-element tiles per CTA; `cake_mhc` retained `dY` (`508.2 MiB`/layer at `T=16,263`) across all backward layers.
9. **TileLang `T.Parallel` Int Coercion, `ReduceLowerer` Out-of-Bounds Shared-Memory Corruption & `*_handle` Restrict (`714cbf150d`, `dbabf135f9`, `a366af801a`)**: When `thread_offset > 0` (e.g., `gemm` on `[0, 128)` and `reduce` on `[32, 160)`), unguarded `run_batch` allowed threads `[0, 32)` to underflow unsigned indexing and corrupt `64` floats (`256 B`) of shared memory preceding the reduce workspace.

---

## 2. Implementation Mechanics & Quantitative Evidence

### 2.1 `cake_kda_train` Plan Cache & Zero-Sync Arena (`bd60c11809`)
- **Plan Cache & Device Arch Selection**: `cake_backend.plan()` caches on `(S, num_docs, H, sm_count, arch)`, cutting repeated `plan()` latency on `S = 261,393` (4 packed docs) from `1.044 ms` (63 `select_record` calls) to **`0.0016 ms`** (0 selections, **652.5x** speedup) and selecting `sm_103a` on GB300 when `arch=None`.
- **Zero Host Synchronizations**: Device `arange` tail-split counts and `non_blocking=True` pinned `doc_lengths` copies reduce host syncs per step from `2–6` to **`0`**, cutting host wait behind a 98 ms GPU spin from `101.45–105.88 ms` to **`1.73–4.33 ms`** on GB200 (`1.68–4.17 ms` on GB300) with zero retained device memory after each step.

### 2.2 Rubin `sm_107a` SSD Combined, GB300 Muse Norm, MM FP4 Multicast & SiLU CTA-Cap
- **Rubin `sm_107a` Mamba2 SSD (`ecd4d71d89`)**: Launches 20 of 56 programs with `__cluster_dims__(1, 1, 1)`, adds 3 `exactpre_{bf16,f16,f32}_varlen_tailone` runtime-ticket flag-word programs, and broadcasts sequence-table entries in `exactpre_f32_varlen_claim`, accelerating 45 of 93 shapes by **`1.004x–1.084x`** (**`1.022x`** geomean).
- **GB300 / B200 `cake_muse_spark_norm` (`d9447704cb`)**: Retunes gated residual RMSNorm (`H=8192`) to **`1.1360x`** in GB300 PDL graph replay (`M=192`, `3.439 -> 3.027 us`) and **`1.1289x`** kernel-alone (`M=512`, `9.847 -> 8.722 us`), while adding a 4-CTA 2-row L2-prefetch next-RMSNorm program on B200 (`1.0478x` at `M=1536`).
- **GB300 `cake_mm_fp4` (`c9fc6cab40`) & Grouped FP8 SiLU CTA-Cap (`a0384efcba`)**: Cluster-of-8 `2x2` multicast (`_sfr_mc42`) yields **`1.038x–1.066x`** across 12 `M = 1793..2048` rows on GB300; `select_route_and_cap` lifts `FUSED_MIXED_ROUTE` to `{144, 148, 152}` CTAs (`148` CTAs on `M=8192`, 16 experts), achieving **`1.064x` graph / `1.073x` eager** on B200 and **`1.032x` / `1.043x`** on GB300.

### 2.3 PDL Coherent Alpha Load, SM120 Weight-First, GDN, Softmax Balanced Chunks & mHC Release
- **PDL Coherent `alpha[0]` Load (`464ca3c813`)**: Replaces non-coherent `alpha[0]` reads with volatile `ld.global.b32` strictly after `griddepcontrol.wait` on `16x1024x1024` `blackwell-native` rows (`1.007x–1.018x`).
- **SM120 Weight-First (`1e7a60d232`) & Deterministic GDN (`4cc027c249`)**: Overlaps weight TMA loads with software activation gather in `fp8-grouped-tc-w8-weight-first-v6`; `cake_gdn_det_train` LPT longest-first dispatch and 2-stream tail fill deliver **`1.289x`** fwd+bwd on B200 (`4x4096`) and **`1.279x`** fwd on GB300 (`3 seqs = 12,750 tok`).
- **Softmax Balanced Chunks (`f4be6e9df5`) & mHC `dY` Release (`1fb63709b8`)**: `ceil(vocab / CTAs)` rounded up to 32 speeds up all 219 changed B200 shapes by **`1.010x–1.094x`** (**`1.045x`** geomean); releasing `dY` immediately after each mHC connection's backward frees **`508.2 MiB` per layer** (**`29.29 GiB`** across 59 of 60 layers at `T=16,263`).

### 2.4 TileLang Compiler Passes (`714cbf150d`, `dbabf135f9`, `a366af801a`)
- **`ReduceLowerer` Thread-Bounds Guard (`dbabf135f9`)**: Guards `run_batch` with `[thread_offset, thread_offset + extent)` (`[32, 160)`), preventing the 32 non-participating threads `[0, 32)` from overwriting **`64` floats (`256 B`)** of shared memory before the reduce workspace.
- **`T.Parallel` & `NonRestrictCollector` (`714cbf150d`, `a366af801a`)**: Coerces Python `int` `coalesced_width` via `tirx.convert` and preserves `__restrict__` on `*_handle` parameters.

---

## 3. Deterministic Verification

- **Verification Script**: `experiments/flashinfer_kda_plan_ssd_sm107_muse_mm_fp4_silu_bf16fp4_sm120_gdn_softmax_mhc_tilelang_phase124/run.py`
