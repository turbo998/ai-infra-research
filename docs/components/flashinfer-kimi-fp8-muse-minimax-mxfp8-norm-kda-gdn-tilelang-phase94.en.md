# FlashInfer Kimi-K3 FP8 Registry Deslop, Muse Spark SM107a PDL Norm, MiniMax-H3 SM103/SM120 Projections, Grouped MXFP8 224-Thread Quantize, Fused Norm-Combine `min.s16x2` Poll, KDA Prefill, SM8x Chunk-Parallel GDN & TileLang Ascend Exact Placement (Phase 94)

> **中文版**: [flashinfer-kimi-fp8-muse-minimax-mxfp8-norm-kda-gdn-tilelang-phase94.md](flashinfer-kimi-fp8-muse-minimax-mxfp8-norm-kda-gdn-tilelang-phase94.md)  
> **Evidence Class**: `implementation` + `author-benchmark` + `own-offline`  
> **Experiment Path**: `experiments/flashinfer_kimi_fp8_muse_minimax_mxfp8_norm_kda_gdn_tilelang_phase94/`

---

## 1. Architectural Context & Core Problems

Phase 94 Track 1 analyzes ten primary upstream commits across FlashInfer (`#7399`, `#7400`, `#7397`, `#7391`, `#7390`, `#7388`, `#7347`, `#5109`) and TileLang (`#3607`, `#3625`):

1. **Kimi-K3 FP8 Projection Launcher Deslop & Decode Bucket Expansion (`#7399`)**: Collapses 54 per-kernel launcher `.cu` files into 5 per-argument-plan launchers (`-90.74%`), folds 5 lever-twin kernels behind `#if` guards (`75 -> 70`), removes dead `_sks` stream-K dispatch, and extends `DECODE_TABLE` M-buckets from `256` to `512/1024/2048`.
2. **Muse Spark Norm SM107a PDL Routing (`#7400`)**: Adds Rubin `sm_107a` (212 SMs) `bf16_pdl` dispatch for `rmsnorm_sigmoid_gate` (`(rows, 64)` and `(rows, heads, 64)`) and retunes plain bandwidth-band thresholds (`h128 >= 384`, `h32 >= 768`, `h16 >= 2048` tokens).
3. **MiniMax-H3 SM103 256-Bit Residual Loads & SM120 Pre-Attention (`#7397`, `#7347`)**: Replaces two 128-bit loads per 16-column BF16 residual chunk with a single 256-bit load and issues first-unit weight loads before `griddepcontrol.wait` on `sm_103a` (`608..3648` tile-pair windows on 76 cluster pairs); accelerates `sm_120a` FP8->BF16 pre-attention by `1.3655x` at `M=33472` (`18.6993 -> 13.6945 ms`).
4. **Grouped MXFP8 224-Thread Row-Tiled Quantization (`#7391`)**: Routes `K = 7168 = 224 x 32` rows with `B*M >= 16384` to a 224-thread (7-warp) single-CTA-per-row kernel, halving mask-blind CTA launches (`65,536 -> 32,768` on `[32, 1024, 7168]`) for a `1.41x–1.60x` sparse-mask speedup.
5. **Fused Norm-Combine Packed `min.s16x2` Readiness Tree (`#7390`)**: Replaces 7 scalar comparisons across 8 peer readiness flags with a 3-instruction `min.s16x2` tree and prefetches workspace control records right after `griddepcontrol.wait` in the 4-CTA pipeline (`8.128 -> 7.712 us`, `1.054x` at `T=1` on B200).
6. **BF16-Input FP32-State KDA Prefill & SM8x 4-Stage Chunk-Parallel GDN (`#7388`, `#5109`)**: Rebuilds 22 KDA prefill modules as 36 programs/arch (`1.16x` at `T=8192` on GB300, `1.22x` at `BS4xT64` on B200); introduces a 4-stage (`T -> MN -> fixup -> prefill`) SM8x Gated Delta Rule pipeline that shrinks the serial recurrence chain at `L=8192` from `128` steps (`64`-token tiles) to `4` chunks (`2048`-token chunks, `32x` reduction) with a `CopyR2SOp` fallback when `stmatrix` is absent.
7. **TileLang Ascend Exact On-Chip Placement & `Int64Promoter` Overlap Analysis (`#3607`, `#3625`)**: Enforces hardware alignment and capacity invariants for `tl.ascend_buffer_addresses` (`UB 248 KiB / 216 KiB with 32 KiB SIMT`, `L1 512 KiB`, `L0A/B 64 KiB`, `L0C 256 KiB`, `SS 3 KiB`) and promotes region bounds `[min, min + extent)` to `int64` prior to `RegionsMayConflict` to eliminate `int32` wraparound at `(1 << 31) - 1`.

---

## 2. Deterministic Verification Summary (`own-offline`)

| Module & PR | Metric | Before | After | Impact |
|---|---|---:|---:|---|
| FlashInfer `#7399` Kimi-K3 FP8 | Launcher `.cu` files / Max decode bucket | `54` / `M<=256` | `5` / `M<=2048` | `-90.74%` launchers |
| FlashInfer `#7397` / `#7347` MiniMax-H3 | SM103 16-col residual loads / SM120 FP8 pre-attn | `2x128b` / `18.6993 ms` | `1x256b` / `13.6945 ms` | `1.3655x` (`M=33472`) |
| FlashInfer `#7391` Grouped MXFP8 | Grid CTAs on `[32, 1024, 7168]` | `65,536` (`128` threads) | `32,768` (`224` threads) | `2.0x` fewer CTAs / `1.60x` faster |
| FlashInfer `#7390` Fused Norm-Combine | B200 `T=1` / `T=8` latency (`H=2560`) | `8.128` / `10.560 us` | `7.712` / `10.048 us` | `1.054x` / `1.051x` |
| FlashInfer `#7388` / `#5109` KDA & GDN | GB300 KDA `T=8192` / GDN serial steps | `275.78 us` / `128` steps | `238.53 us` / `4` steps | `1.16x` / `32x` shorter chain |
| TileLang `#3607` / `#3625` Ascend | Available UB with SIMT / `int64` endpoint | `248 KiB` / wrapped `-2^31` | `216 KiB` / `2147483648` | Exact layout & barrier safety |
