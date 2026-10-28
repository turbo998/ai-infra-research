# FlashInfer GB300 Separate SF-Ring FP4 GEMM, CAKE 27 Paged Decode Attention Programs, Rubin MLA Delayed PDL Trigger & Dual Merge, SVDQuant One-Wave Tail Build, MiniMax H3 & Warp Decode, B12X MoE Power-of-Two Workspace Bucketing, cuDNN 1.32 QSA & TileLang ROCm 10 Architecture Analysis (Phase 120)

> **Module Positioning**: Written for cloud Solution Architects mastering Blackwell Ultra (`sm_103a`) and Rubin (`sm_107a`) kernel microarchitecture. Based on 10 upstream commits landed in `flashinfer-ai/flashinfer` and `tile-ai/tilelang` between 2026-10-27 and 2026-10-28, this document dissects decoupled scale-factor TMA rings, paged decode AOT kernel families, asymmetric tail-wave repacking, and JIT workspace cache bucketing.
> **Evidence Class**: `implementation` (verified at fixed commit SHAs) + `author-benchmark` (upstream PR measurements, not re-measured on project GPUs) + `own-offline` (reproducible Python standard-library model in this repository).

---

## 1. Architectural Problem Statement & 10 Upstream Commits Overview

On Blackwell (`sm_100a`, 148/152 SMs), Blackwell Ultra (`sm_103a`, GB300 152 SMs), and Rubin (`sm_107a`, 208 SMs), ultra-dense FP4/FP8 Tensor Core throughput shifts kernel bottlenecks from main MMA loops to:
1. **Shared TMA pipeline contention between A/B operands and block scale factors (SF), plus shared-memory epilogue round-trips**;
2. **Low CTA occupancy on tail waves across 76 2-CTA pairs (152 SMs)**;
3. **Repeated `cuda.tile` / CuteDSL recompilations whenever dynamic MoE `routed_rows` fluctuates by a single token**.

| # | Repository & Commit | Date | Core Mechanism & Architectural Change | Evidence Class |
| :--- | :--- | :--- | :--- | :--- |
| A1 | `flashinfer-ai/flashinfer` [`4ca4e6e`](https://github.com/flashinfer-ai/flashinfer/commit/4ca4e6e485838462334de5aa9012f3895354341f) | 2026-10-28 | `cake_mm_fp4`: GB300 (`sm_103a`) 6-stage A/B ring + separate 5-stage SF ring and register-direct epilogue (`gemm_2cta_ptr`) | `implementation` + `author-benchmark` |
| A2 | `flashinfer-ai/flashinfer` [`5e0e99b`](https://github.com/flashinfer-ai/flashinfer/commit/5e0e99bd2e8be8f3ebdb3a3a0bd670ab114c425f) | 2026-10-28 | `cake_fmha`: 27 AOT paged decode attention programs (`BF16`, `FP8`, `NVFP4 KV`, `head_dim=256` small-M) for `sm_100a`/`sm_103a` | `implementation` + `author-benchmark` |
| A3 | `flashinfer-ai/flashinfer` [`f6587bc`](https://github.com/flashinfer-ai/flashinfer/commit/f6587bc5dd490e1b08a18040ab5fb6496b599603) | 2026-10-28 | `cake_rubin_mla_nvfp4_decode`: Rubin `sm_107a` later PDL trigger (`items <= units + pairs`) and `probe_after_loads` / `column` merge | `implementation` + `author-benchmark` |
| A4 | `flashinfer-ai/flashinfer` [`03ebbc6`](https://github.com/flashinfer-ai/flashinfer/commit/03ebbc6e7121749d802991553b4093383431c154) | 2026-10-28 | `cake_nvfp4_svdquant`: `sm_100a` `(4096, 3072, 12288, rank=32)` 40-tile tail repacked into one 76-unit wave (`8×192 + 68×128`) | `implementation` + `author-benchmark` |
| A5 | `flashinfer-ai/flashinfer` [`d949946`](https://github.com/flashinfer-ai/flashinfer/commit/d949946f67275c61ab977a7806c9f6e806469da6) | 2026-10-28 | `cake_minimax_h3_varlen_attention`: dual ticket mailboxes (words 6 & 7), `quantize_early_qk`, and tiny BF16 plans | `implementation` + `author-benchmark` |
| A6 | `flashinfer-ai/flashinfer` [`7de971a`](https://github.com/flashinfer-ai/flashinfer/commit/7de971a26c89340e12d4298f9dfbd4a48da28e50) | 2026-10-27 | `cake_warp_decode`: count-rank route-pack head (129 SM100 + 10 SM103 rows) and early-load finalizers (46 SM100 + 12 SM103 rows) | `implementation` + `author-benchmark` |
| A7 | `flashinfer-ai/flashinfer` [`959a113`](https://github.com/flashinfer-ai/flashinfer/commit/959a113cefa2a8e43c924b4dc411f165164df2c5) | 2026-10-28 | `b12x`: power-of-two MoE workspace row bucketing (`1 << (max(128, R)-1).bit_length()`) to stop kernel-cache thrash | `implementation` |
| A8 | `flashinfer-ai/flashinfer` [`c2ba71f`](https://github.com/flashinfer-ai/flashinfer/commit/c2ba71f72c38f2ff29ec325dfcc8ba7f35743ef0) | 2026-10-28 | `cudnn`: cuDNN 1.32 native QSA forward APIs (`qsa_forward` uint8 bitmap & `qsa_indexed_forward` int32 token indices) | `implementation` |
| A9 | `tile-ai/tilelang` [`040b305`](https://github.com/tile-ai/tilelang/commit/040b3054f1c20b1514be0b8e497ccbb791009986) | 2026-10-27 | `target/utils.py`: ROCm `auto` target queries `torch.cuda.current_device()` instead of hardcoded device `0` | `implementation` |
| A10 | `tile-ai/tilelang` [`015caa2`](https://github.com/tile-ai/tilelang/commit/015caa2b6e351e91d3b54922b9d130a8ec3b99f0) | 2026-10-27 | `env.py` / `utils.py`: pip-SDK ROCm 10 (`rocm-sdk`) 4-tier arch detection and 10s `rocminfo` timeout fallback | `implementation` |

---

## 2. Deep Technical Analysis & Formulations

### 2.1 GB300 (`sm_103a`) Separate Scale-Factor Ring & Register-Direct Epilogue (`4ca4e6e`)

In `flashinfer/cake/kernels/mm_fp4/program.py` targeting `SF_RING_ARCH = "sm_103a"` (GB300, 152 SMs):
- **Decoupled TMA Rings**: While `sm_100a` K256 programs couple A/B FP4 tiles and E4M3 scale factors into a single 4-stage ring, GB300's larger shared memory allows `NUM_TMA_PIPE_STAGES = 6` for A/B tiles alongside an independent `NUM_SF_PIPE_STAGES = 5` scale-factor ring with dedicated `mbarrier` tracking.
- **Register-Direct Epilogue**: In `gemm_2cta_ptr`, TMEM accumulators are read into registers, converted to BF16, and stored directly to global memory without staging through shared memory and TMA store descriptors.
- **Author-Benchmarked Gains (`author-benchmark`)**:
  - **1.035x–1.107x** across 16 DeepSeek/Llama GEMM shapes (`16384×7168×2048` drops from `62.33 us` to `56.35 us`, **1.1061x**);
  - **1.030x–1.084x** across 14 `quant+GEMM` chain shapes (`28672×8192×2048` drops from `156.53 us` to `144.46 us`, **1.0836x**).

### 2.2 CAKE FMHA 27 Paged Decode Programs & Device-Scale Pointer Launch (`5e0e99b`)

`cake_fmha` adds **27 AOT paged decode programs** (19 compiled per target across `sm_100a` and `sm_103a`, covering `D=128/256`, GQA ratios `4/5/8/16`, KV page sizes `1/16/32/64`, and `BF16` / `FP8` / `NVFP4 KV`):
- **Head-Dim 256 Small-M Path**: Activates `head_dim_256_small_m` when `Q_len * group_size <= 8`.
- **Device-Side Scale Pointer**: Passing GPU-resident `q_scale` / `kv_scale` pointers directly into the kernel eliminates host `.item()` synchronization, cutting B200 FP8 (`B=128, kv=4096`) latency from `59999.70 us` to `169.89 us` (**353.17x**).
- **B200 Decode Performance (`author-benchmark`)**: BF16 `B=4, kv=1024, gqa=8, D=128` improves from `13.44 us` to `7.68 us` (**1.750x**); FP8 `B=256, kv=8192` reaches `317.31 us` vs `trtllm-gen` `321.45 us`.

### 2.3 Rubin (`sm_107a`) MLA NVFP4 Decode Delayed PDL & SVDQuant One-Wave Tail (`f6587bc`, `03ebbc6`)

1. **Rubin `sm_107a` MLA NVFP4 Decode (`f6587bc`)**:
   - Widens the Programmatic Dependent Launch (PDL) trigger threshold from `items <= units` to $\text{items} \le \text{units} + \text{pairs}$ (`50 + 4 = 54` on a 208-SM Rubin board);
   - Introduces `probe_after_loads` and `column` (4-warp, `384–448` rows, `17–32` splits, `8 Bytes` packed load per split), achieving up to **1.067x** on `H192-Q1-B2-KV128K`.
2. **SVDQuant One-Wave Tail Build (`03ebbc6`)**:
   - For `(M=4096, N=3072, K=12288, rank=32)` on 152 SMs (76 CTA pairs), 192 `256×256` tiles leave a 40-tile tail ($40 \times 256 = 10240$ cols) after 2 full waves (152 tiles);
   - Repacks those 40 tiles into exactly **76 work units**: 8 pieces of `192` cols and 68 half-tiles of `128` cols ($8 \times 192 + 68 \times 128 = 10240$ cols, $8 + 68 = 76$ CTA pairs), reducing GB200 latency from `62.98 us` to `59.17 us` (**1.0644x**).

### 2.4 B12X MoE Workspace Bucketing, cuDNN 1.32 QSA & TileLang ROCm 10 (`959a113`, `c2ba71f`, `040b305`, `015caa2`)

- **B12X MoE Power-of-Two Bucketing (`959a113`)**: `_bucket_workspace_rows(R) = 1 << (max(128, int(R)) - 1).bit_length()` bounds `cuda.tile` static-backend recompilations from 238 distinct shapes (`64..4096`, step 17) down to **6 buckets** (**39.67x** reduction), with a safe fallback when total elements exceed $2^{31}-1$.
- **cuDNN 1.32 QSA Forward APIs (`c2ba71f`)**: Adds `qsa_forward` (`uint8` 4-token block bitmaps, 512/513 token kernel boundary) and `qsa_indexed_forward` (`int32` indices).
- **TileLang ROCm Multi-Device & ROCm 10 pip-SDK (`040b305`, `015caa2`)**: Queries `torch.cuda.current_device()` instead of hardcoded device `0` and establishes a 4-tier architecture lookup with a 10s `rocminfo` timeout so pip-installed ROCm 10 never silently miscompiles for `gfx900`.

---

## 3. Offline Reproducible Verification (`own-offline`)

Experiment directory: `experiments/flashinfer_gb300_sf_ring_cake_fmha_rubin_mla_svdquant_tilelang_phase120/`

- **Command**: `python experiments/flashinfer_gb300_sf_ring_cake_fmha_rubin_mla_svdquant_tilelang_phase120/run.py --check`
- **Verified Outputs**:
  - `gb300_sf_ring_fp4_gemm.max_gemm_speedup = 1.1061`, `max_chain_speedup = 1.0836`;
  - `cake_fmha_paged_decode.bf16_b4_kv1024_speedup = 1.75`, `device_scale_speedup = 353.17`;
  - `rubin_mla_and_svdquant.svdquant_single_wave_fit = true` (`8*192 + 68*128 = 10240`, `76` units);
  - `b12x_qsa_tilelang_rocm.b12x_compile_reduction_ratio = 39.67` (`6` buckets across 238 shapes).

---

## 4. Architect Design Exercise & Falsifiable Questions

1. **Design Exercise**: On a 148-SM B200 (74 2-CTA pairs), `(M=4096, N=3072)` produces 192 `256×256` tiles: 2 full waves (148 tiles) + 44 tail tiles ($11264$ cols). Solve $a \cdot 192 + b \cdot 128 = 11264$ with $a + b = 74$: $a = 28$ pieces of 192 cols and $b = 46$ half-tiles of 128 cols.
2. **Falsifiable Question**: When `routed_rows` sits just above a power-of-two boundary (e.g., `R = 2049` rounded to `4096`), under what batch regime does zeroing/allocating the padded workspace rows cost more than the saved JIT lookup overhead?
