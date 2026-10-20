# FlashMLA Blackwell SM100 DeepSeek-V4.1 Fused `Norm+RoPE+Attn+RoPE+Cast` Kernel & Bank-Conflict-Free `KVBlockDequantizer`

> **中文版**: [flashmla-sm100-v41-fused-norm-rope-attn-dequant-kv.md](flashmla-sm100-v41-fused-norm-rope-attn-dequant-kv.md)  
> **Evidence Class**: `implementation` (First-hand source verification) + `own-offline` (Deterministic stdlib simulation)  
> **Upstream Repository & Commits**: `deepseek-ai/FlashMLA` commits [`07a1089857b63e74e3133630c02b083b75e8d4b2`](https://github.com/deepseek-ai/FlashMLA/commit/07a1089857b63e74e3133630c02b083b75e8d4b2) (2026-09-10) & [`ba89a3466e9470ad08ab39738d4e7bb66989e1e7`](https://github.com/deepseek-ai/FlashMLA/commit/ba89a3466e9470ad08ab39738d4e7bb66989e1e7) (2026-09-15)  
> **Experiment**: [`experiments/flashmla_sm100_v41_fused_dequant_sim/run.py`](../../experiments/flashmla_sm100_v41_fused_dequant_sim/run.py) (`--check` passed)

---

## 1. Architectural Motivation & Engineering Bottlenecks

In DeepSeek-V4 / V4.1 (`d_qk = d_v = 512, h_kv = 1, d_rope = 64`) token-level exact top-k sparse attention, launching separate kernels for `RMSNorm`, `Q RoPE`, `Core Attention`, `O RoPE`, and `O FP8 Quantization` across the `q_b_proj -> wv_proj` segment incurs repeated HBM round-trips and kernel launch latency. Furthermore, during decoding over mixed `V41` (FP8) primary and `V41_FP4` (FP4 E2M1 + E4M3 scale) secondary KV caches, in-place shared-memory dequantization of TMA `gather4` tiles faces 4-way shared-memory bank conflicts and `ptxas` live-range merging across formats.

---

## 2. First-Hand Source Analysis (`deepseek-ai/FlashMLA`)

### 2.1 Four-Generation Paged Quantized KV Cache Formats & `TMA_K_STRIDE >= 256` (`csrc/kernels/kv_cache_format.h`)

`KVCacheFormat<MT>` stores each paged block as two row-major byte regions (`[page_block_size, TMA_K_STRIDE]` quantized token data and `[page_block_size, NUM_SCALES_EACH_TOKEN * SCALE_BYTES]` scale factors):

| `ModelType` | `D_QK` | `D_FP8` / `D_FP4` / `D_BF16` | `QUANT_TILE_SIZE` | `NUM_SCALES` × `SCALE_BYTES` | `TMA_K_STRIDE` | `BYTES_PER_TOKEN` | Compression vs V3.2 |
|---|---:|---|---:|---|---:|---:|---:|
| `V32` | 576 | `512 / 0 / 64` | 128 | `4 × 4B (inline fp32)` | `656 B` | **`656 B`** | `1.0000x` |
| `V4` | 512 | `448 / 0 / 64` | 64 | `8 (7+1 pad) × 1B (ue8m0)` | `576 B` | **`584 B`** | `1.1233x` |
| `V41` | 512 | `512 / 0 / 0` | 32 | `16 × 1B (ue8m0)` | `512 B` | **`528 B`** | `1.2424x` |
| `V41_FP4` | 512 | `0 / 512 / 0` | 16 | `32 × 1B (e4m3)` | `256 B` | **`288 B`** | **`2.2778x`** |

- **`TMA_K_STRIDE >= 256` Hardware Invariant**: `V41_FP4` has `QUANT_BYTES = 512 / 2 = 256 B` (`TMA_K_STRIDE = 256 B`), meeting the minimum 256 B row stride required for `int32` TMA coordinates to index a full KV cache while achieving **`1.8333x`** compression over `V41` FP8 (`528 B`).

### 2.2 Bank-Conflict-Free Warpgroup Dequantization & Bit-Exact `e2m1 × e4m3` (`csrc/kernels/sm100/dequant_utils.cuh`, `helpers.h`)

1. **`for_each_kv_block<Orig, Extra>` Loop Separation**: When `Orig` (`V41`) and `Extra` (`V41_FP4`) differ, blocks `[begin, min(num_orig_blocks, end))` and `[max(begin, num_orig_blocks), end)` execute in two disjoint `CUTE_NO_UNROLL` loops so `ptxas` never merges register live ranges of both formats.
2. **`KVBlockDequantizer` Quarter-Bank Stride Padding (`RAW_TOKEN_SMEM_STRIDE / 4 % 16 == 8`)**:
   - A 128-thread warpgroup is partitioned into `NUM_GROUPS = 16` groups of `GROUP_SIZE = 8` threads (`ELEMS_PER_STEP = 64`).
   - Without padding, 1-CTA (`256 B`) or 2-CTA (`128 B`) FP4 rows satisfy `stride / 4 % 16 == 0`, causing the 4 rows of a `gather4` group to collide on the exact same shared-memory banks (**`4-way` bank conflict** on `LDS.32`).
   - Setting the TMA box `+32 B` wider (`288 B` for 1-CTA, `160 B` for 2-CTA; zero-filled by TMA) satisfies `RAW_TOKEN_SMEM_STRIDE / 4 % 16 == 8`, staggering the 4 rows across the 4 quarters of the 32 banks (banks 0, 8, 16, 24) for **`1-way` conflict-free `LDS.32`**.
   - In the `LDS.128` common dequant path of `fused_norm_rope_attn_rope_cast_fwd`, padding each row by `+16 B` (`272 B / 144 B` for FP4, `528 B / 272 B` for V41 FP8) makes `RAW_TOKEN_SMEM_STRIDE / 16` odd (`17 / 9 / 33 / 17`), eliminating `LDS.128` bank conflicts.
3. **`SW128` K-Major XOR Swizzle & Lossless `e2m1 × e4m3` in BF16**:
   - `dst_offset = (group_idx / 8) * 1024 + row_in_atom * 128 + (idx_in_group ^ row_in_atom) * 16` XORs the 16 B lane index with `row_in_atom = group_idx % 8`, guaranteeing conflict-free 8-lane wavefront `STS.128`.
   - Because `e2m1` has at most 2 significant mantissa bits and `e4m3` has at most 4, their product has at most `2 + 4 = 6` significant bits, fitting losslessly inside BF16's 8-bit significand (`0` rounding error across all `2,032` finite non-NaN pairs).

### 2.3 SM100 Fused `Norm+RoPE+Attn+RoPE+Cast` TMEM Packing, Weight Permutations & `st.async` LLP64 Fix

- **TMEM 512-Column Packing (`core_attn/config.h:tmem_cols`)**:
  - `H_Q = 64` (`CLUSTER_SIZE = 1, FOLD_FACTOR = 2, B_TOPK = 64, NUM_MRGEMM_RAILS = 2, NUM_P_BUFS = 2`): `O = [0, 256)`, `Q = [256, 384)`, `P_0 = [384, 448)`, `P_1 = [448, 512)` -> exactly **`512` columns** with `NEED_TP_EMPTY_BAR = 0` (`P[i] <- O[i-2] <- S[i-2] <- P[i-2] free`).
  - `H_Q = 128` (`CLUSTER_SIZE = 2, FOLD_FACTOR = 2, B_TOPK = 96, NUM_P_BUFS = 1`): `O = [0, 256)`, `Q = [256, 384)`, `P_0 = [384, 480)` -> **`480` columns** (`<= 512`, `NEED_TP_EMPTY_BAR = 1`).
- **Pre-Permuted Weight Bijections (`permute_q_b_proj`, `permute_wv_proj`)**:
  - `permute_q_b_proj_kernel` maps `row_idx = head_idx * 512 + head_dim_idx` to `out_row_idx = (head_dim_idx % 16) + head_idx * 16 + (head_dim_idx // 16) * (h_q * 16)` (bijective across all `65,536` rows).
  - `permute_wv_proj_kernel` transposes 32-element chunks `(head_idx, head_dim_chunk_idx)` within each `wv_group` (`wv_group_size = 8, D_O = 512`) to `head_dim_chunk_idx * 8 + head_idx` (`128` chunks), matching `out_fp8` without runtime transpose.
- **128-bit `st.async` MSVC LLP64 Fix (`ba89a3466e`, `intrinsics.cuh`)**: Replaces `long2` (`8 B` on Windows LLP64) with `longlong2` (`16 B`), preventing silent upper-64-bit truncation in `st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v2.s64`.

---

## 3. Deterministic Offline Verification (`flashmla_sm100_v41_fused_dequant_sim`)

Running `python experiments/flashmla_sm100_v41_fused_dequant_sim/run.py --check` verifies:

| Verification Dimension | Baseline / Unpadded | FlashMLA SM100 V4.1 Design | Verified Outcome |
|---|---|---|---|
| KV Cache Bytes / Token | `V32`: `656 B` / `V4`: `584 B` / `V41`: `528 B` | `V41_FP4`: **`288 B`** (`TMA_K_STRIDE = 256 B`) | **`2.2778x`** (vs V3.2) / **`1.8333x`** (vs V41 FP8) |
| `KVBlockDequantizer` `LDS.32` Bank Way | Unpadded (`256 B` / `128 B`): **`4-way`** conflict | `+32 B` padded (`288 B` / `160 B`): **`1-way`** | **Eliminates 4-way SMEM bank conflicts** |
| `e2m1 × e4m3 -> bf16` Exactness | 2,032 finite non-NaN pairs | Low 16 bits of FP32 product `== 0x0000` (`inexact = 0`) | **100% bit-exact (`0` rounding error)** |
| Fused Kernel TMEM Columns | `H_Q=64` (`B_TOPK=64`, 2 `P` bufs) / `H_Q=128` (`B_TOPK=96`) | **`512` cols** (`NEED_TP_EMPTY_BAR=False`) / **`480` cols** | Fits strictly within SM100 `512`-col TMEM |
| Weight Permutations & `st.async` | `permute_q_b_proj` (`65,536` rows) / `permute_wv_proj` (`128` chunks) | Bijective `True`; `longlong2 = 16 B` vs `long2 = 8 B` | Zero runtime transpose + LLP64-safe 128b store |
