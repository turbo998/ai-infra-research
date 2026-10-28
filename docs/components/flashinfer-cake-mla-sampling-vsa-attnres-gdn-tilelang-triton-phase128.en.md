# Phase 128 Deep Dive (Track A): FlashInfer Cake NVFP4 MLA Setup Prefetch, Sampling Round 17 Whole-CTA Tails & Compact Lists, Hopper VSA Position Pinning, Kimi-K3 Parked Accumulators, GDN No-Sync Prepare & TileLang/Triton Contracts

## 1. Motivation & Core Problem

Across Blackwell (`sm_100a`/`sm_103a`/`sm_120a`) and Hopper (`sm_90a`) GPU software stacks, microsecond-scale kernels are bottlenecked by **post-rendezvous first-item metadata latency**, **two-launch top-k tail overhead**, **register spills in high-K attention-residual pairs**, and **host-synchronizing context preparation and kept forward tapes in context-parallel GDN training**. This deep dive analyzes 11 upstream commits merged on 2026-10-28 (`flashinfer-ai/flashinfer` × 9, `tile-ai/tilelang` × 1, `triton-lang/triton` × 1) and validates their arithmetic and resource invariants.

## 2. Key Implementation Paths & Architectural Mechanisms

### 2.1 Cake NVFP4 MLA Short-Unit Decode Setup-Time First-Item Load (`949a9f2d53ca`)
- **Path**: `flashinfer/experimental/nvfp4_mla_decode/{cake_backend.py,cake_jit.py}` adds `decode_short_spec_uh` (SM100) and `decode_short_epi_warp_spec_uh` (SM100/SM103).
- **Mechanism**: When every work unit in a decode plan holds exactly one item pair, item `2 * cluster` is guaranteed to be each unit's first item and resides inside the metadata table. The new programs preload item `2 * cluster` during kernel setup ahead of cluster rendezvous and start the load warp immediately, achieving up to **`1.026x`** speedup on B200 and **`1.029x`** on GB200.

### 2.2 Cake Sampling Round 17 Whole-CTA Tails & SM12x Compact-List Streams (`3812cddafdb7`)
- **Path**: `csrc/cake_sampling/cake_sampling_binding.cuh` adds `kFlagCompactList = 8192`, expanding stage-1 builds from `108 -> 135` (total kernels `123 -> 150`).
- **Mechanism**:
  1. One-launch leader-push and default-sample whole-CTA tails for `(8, 16)`, `(1, 16)`, and `(2, 32)` streams eliminate the two-launch chain at large `top_k > 768..800` (up to **`1.125x`** on B200, **`1.121x`** on H100, **`1.119x`** on GB300).
  2. Twenty-three compact-list twins (`8,192` candidate entries per CTA) fit inside the `101,376 B` shared-memory opt-in of SM12x (RTX PRO 6000, 188 SMs), delivering **`1.307x–1.413x`** speedup at vocab `128,256 / 151,936` and batch `32..128`.

### 2.3 Hopper Cake VSA Position-Selected Sage-FP8 GQA Static Program (`279f9a082fb2`)
- **Path**: `flashinfer/cake_vsa_sm90_backend/cake_planner.py` adds `PIN_MIN_CTA_LOOP_ITERS = 14` and `attention_fp8_gqa_pin`.
- **Mechanism**: Static Sage-FP8 GQA plans whose lightest CTA executes `cta_loop_iters >= 14` route to `attention_fp8_gqa_pin`, which packs E4M3 `P` tiles via 16-bit carriers (`mov.b32 {lo, hi}`) and pins the V-ring release barrier 6 instructions after V wait using a bit-identical `mbarrier.test_wait` + `min(bias, fma(token, 1e30, bias))` dependency gate rather than letting `ptxas` sink it behind 66 `ex2` instructions. Main-loop instructions drop from `675 -> 633` (`-42`), delivering **`1.0287x`** geomean speedup across 20 routed shapes (up to **`1.0361x`**) with 168 registers and zero spills.

### 2.4 Cake Kimi-K3 AttnRes Parked Accumulators & FP8 Projection GEMM Twins (`d0ac3f0b1649`, `21e02addf9f6`)
- **AttnRes Parked Accumulators (`d0ac3f0b1649`)**: For `M=192, K=6..8`, 384-thread pair programs reduce consumer warp registers from `232 -> 200` (producer `40 -> 104`) and park each consumer stream's FP32 accumulators in 64 additional Tensor Memory columns (`256 -> 512` TMEM columns/stream) during the first prefix pass, eliminating all register spills and improving kernel latency by up to **`+9.9%`** on B200 and GB300 (`+7.9%` on B300).
- **Kimi-K3 FP8 Projection (`21e02addf9f6`)**: Adds a 2-stage 4-slot GEMM twin where the MMA warp allocates TMEM and frees it immediately upon epilogue release, plus SM103a 352-wide `q_proj` TMA-store routing with L2 `evict_first`/`evict_last` quantization hints, yielding **`1.042x`** (`TP1 kv_b M=256`), **`1.038x`** (`TP8 f_b M=4096`), and **`1.027x`** (`TP1 q_proj M=512`) on GB300.

### 2.5 Cake GDN Training Adjoint Prefetch, No-Sync Prepare & Auto Tape Reuse (`b6089e61ae3f`, `025f5acf562f`)
- **Adjoint-Walk Prefetch & LPT Plan (`b6089e61ae3f`)**: Prefetches two 256-bit `dv/i` carriers one step ahead with a 2-slot shared-memory chunk-table ring (`82,944 -> 83,968 B` dynamic SMEM) and switches the host planner to longest-processing-time (`row_cost_lpt`) segment selection (`1.04x` on GB300 CP4 zigzag).
- **No-Sync Prepare & Auto Tape Reuse (`025f5acf562f`)**: Replaces 24 pageable synchronizing tensor copies in `build_kcp_context` + `prepare` with a single pinned buffer per dtype and non-blocking `copy_`, reducing `(2, 6)` prepare latency behind queued GPU work from **`147.0 ms` to `5.8 ms` (`25.34x` speedup, `24 -> 0` syncs)**. Defaulting `reuse_recompute_tapes="auto"` hands forward tapes over only when GDN is the checkpoint region's final saver, cutting backward peak memory from **`9,251 MiB` to `6,927 MiB` (`2,324 MiB` saved)** and tightening workspace sizing from `3,885 MiB` to `2,894 MiB`.

### 2.6 MoE EP Prims-TS & DeepSeek FP8 Sender Quant, Cake W4A16 Gate & Compiler Fixes (`9eb51e64a872`, `a2b15151fe59`, `714cbf150d90`, `8afd743d38de`)
- **MoE EP DeepSeek FP8 Dispatch (`9eb51e64a872`)**: Enables `PrimsTsConfig` in `MoEEpSplitLayer` and quantizes 7168-dim activations to DeepSeek FP8 (`1x128` blocks: `7,168 B` payload + `224 B` FP32 scales = `7,392 B` vs `14,336 B` BF16) on the sender rank before All-to-All, cutting dispatch payload by **`48.44%`**.
- **Cake W4A16 CUDA 13.0 Gate (`a2b15151fe59`), TileLang `IntImm` Coercion (`714cbf150d90`), and Triton `gfx1250-strict` Block32 Upcast (`8afd743d38de`)**: Enforces CUDA 13.0 for Cake W4A16 kernels, wraps raw Python `int` `coalesced_width` via `tirx.convert` in TileLang `T.Parallel`, and enables hardware Block32 `v_cvt_scale_pk8` upcast on AMD `gfx1250-strict` when lane pairs `n` and `n+16` share scales.

## 3. Deterministic Verification Summary

| Subsystem / Commit | Metric | Baseline | New Path | Gain / Impact |
|---|---|---:|---:|---|
| Cake NVFP4 MLA (`949a9f2d53ca`) | Short-unit decode latency | Post-rendezvous load | Setup-time `2*cluster` load | **`1.026x`** B200 / **`1.029x`** GB200 |
| Cake Sampling R17 (`3812cddafdb7`) | RTX PRO 6000 / B200 sampling | Two-launch / standard list | 8192 compact-list / whole-CTA tail | **`1.413x`** RTX PRO 6000 / **`1.125x`** B200 |
| Cake VSA SM90 (`279f9a082fb2`) | Sage-FP8 GQA main-loop insts | `675` insts | `633` insts (`cta_loop_iters>=14`) | **`1.0287x`** geomean (`1.0361x` max) |
| Cake AttnRes (`d0ac3f0b1649`) | `M=192, K=6..8` regs & TMEM | 232 regs / 256 TMEM cols (spills) | 200 regs / 512 TMEM cols (0 spills) | Up to **`+9.9%`** on B200 & GB300 |
| Cake GDN Train (`025f5acf562f`) | Queued prepare / Bwd peak HBM | `147.0 ms` (24 syncs) / `9,251 MiB` | `5.8 ms` (0 syncs) / `6,927 MiB` | **`25.34x`** prepare / **`-2,324 MiB`** HBM |
| MoE EP Split (`9eb51e64a872`) | 7168-dim token dispatch bytes | `14,336 B` (BF16) | `7,392 B` (DeepSeek FP8) | **`-48.44%`** All-to-All payload |
