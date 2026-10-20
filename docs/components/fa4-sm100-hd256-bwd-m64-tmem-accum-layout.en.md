# FlashAttention-4 Blackwell SM100 `hdim=256` 2-CTA Backward: `M=64` TMEM Accumulator Layout, Zero-Buffer DSMEM `dS` Exchange & `AccumTileOrder` Unified Postprocess

> **中文版本**: [fa4-sm100-hd256-bwd-m64-tmem-accum-layout.md](fa4-sm100-hd256-bwd-m64-tmem-accum-layout.md) | **Back to Index**: [README.en.md](../../README.en.md) | [README.md](../../README.md)

## 1. Engineering Problem: Why `hdim=256` Cannot Fit in SM100 TMEM with an `M=128` 128-Row KV Tile

On NVIDIA Blackwell SM100 (`512` Tensor Memory / TMEM columns per SM, `128 lanes × 32-bit` per column; `227 KiB = 232,448 B` maximum shared memory per SM), the FlashAttention-4 backward pass maintains five FP32 accumulators: `S`, `dP`, `dV`, `dK`, and `dQ`.

Per `Dao-AILab/flash-attention` commits `2b00d76f222c842b245fd98847223c898bc476db` and `965c9212e6be33c821ee9f2b461564c5c91a2503` (`flash_attn/cute/flash_bwd_sm100.py` L265–319):
- Under the standard **`M=128` TMEM layout** (`tile_m = 128, tile_n = 128, acc_m64 = False, lane_rows = 1`), each TMEM lane holds `1` row and each accumulator consumes as many TMEM columns as its logical width:
  $$\mathrm{cols}(dV) = d_v = 256,\quad \mathrm{cols}(dK) = d = 256,\quad \mathrm{cols}(S) = M = 128,\quad \mathrm{cols}(dP) = M = 128,\quad \mathrm{cols}(dQ) = \frac{d}{\mathrm{cta\_group\_size}}$$
  - In 1-CTA mode, `tmem_end = 128 (S) + 256 (dV) + 128 (dP) + 256 (dK) = 768 > 512` columns (**50% overflow**);
  - Even under the 2-CTA `TWO_CTA_SERIAL` schedule with `tile_n = 128` (`lane_rows = 1`), the persistent `dV (256)` + `dK (256)` accumulators alone consume all `512` columns (`tmem_S_offset = 512, tmem_end = 640 > 512`), leaving zero columns for `S`, `dP`, and `dQ`.

---

## 2. First-Hand Source Walkthrough: `M=64` 2-CTA TMEM Layout & Zero-Buffer DSMEM `dS` Exchange

### 2.1 The `M=64` Accumulator Layout (`tile_n = 64, acc_m64 = True, lane_rows = 2`)

`FlashAttentionBackwardSm100` (`flash_bwd_sm100.py` L103–149 `NOTE [M=64 accumulator layout]` and L265–290) resolves this by shrinking the per-CTA KV tile to **`tile_n = 64`** and activating the **2-CTA `M=64` TMEM layout** (`self.acc_m64 = self.use_2cta_instrs and self.tile_n == 64`). Because the `128` TMEM lanes fold two `64`-row halves (`lane_rows = 2`: lanes `0..63` hold columns `[0, N/2)` and lanes `64..127` hold columns `[N/2, N)`), **every accumulator's TMEM column footprint halves to `N // 2`**:

| Accumulator / Operand | Logical Shape `(rows, cols)` | `M=128` (`tile_n=128`) Cols | `M=64` (`tile_n=64, acc_m64=True`) Cols | `TWO_CTA_SERIAL` TMEM Range `[start, end)` |
|---|---|---|---|---|
| **`dV`** | `(64, 256)` | `256` | **`128`** (`256 // 2`) | **`[0, 128)`** |
| **`dK`** | `(64, 256)` | `256` | **`128`** (`256 // 2`) | **`[128, 256)`** |
| **`S^T`** (`Q @ K^T`) | `(64, 128)` | `128` | **`64`** (`128 // 2`) | **`[256, 320)`** |
| **`dQ`** (`dS @ K`) | `(64, 256)` / CTA | `128` | **`128`** (`256 // 2`) | **`[384, 512)`** |
| **`dP^T`** (`dO @ V^T`) | `(64, 128)` | `128` | **`64`** (`128 // 2`) | **`[448, 512)`** (overlaps tail of `dQ`) |
| **`P^T` / `dS^T`** | `(64, 128)` | `64` / `64` (TMEM) | **`0`** (staged in SMEM `sP` / `sdS_dK`) | `0` TMEM cols (served as SMEM `A` operands of `dV`/`dK` MMAs) |

As a result, `tmem_end = 512 <= 512` with a `64`-column unaliased gap (`[320, 384)`) between `S` and `dQ`, enabling `hdim=256` inside the unified SM100 backward kernel.

### 2.2 Why Staging `P / dS^T` in SMEM Eliminates `sdS_xchg` and Its Alias Barrier

In 2-CTA backward execution, `S`, `dP`, `dV`, and `dK` are partitioned by KV rows (`64` KV rows × `128` Q columns per CTA), whereas the 2-CTA `M=128` `dQ = dS @ K` MMA is partitioned by **Q rows** (`64` Q rows × `64` KV columns per CTA). Thus the two CTAs in the cluster must exchange half of `dS` over Distributed Shared Memory (DSMEM).

Under `M=128`, `dS^T` lives in TMEM and requires a dedicated SMEM exchange buffer `sdS_xchg` (aliased onto `sdQaccum`). Under `M=64` (`acc_m64 = True`, `flash_bwd_sm100.py` L483–539 `_check_dS_exchange_layouts`):
1. `dS^T` is written directly to SMEM buffer `sdS_dK` (`(kv=64, q=128)` K-major, identical to `(q=128, kv=64)` M-major);
2. Splitting along Q by `cta_group_size = 2` makes `sdS_dK[None, None, (None, peer_cta_rank)]` a **single contiguous `64 × 64` sub-tensor** with the exact swizzle and linear element order of the peer CTA's `sdS` (`dQ` MMA `A` operand);
3. Consequently, `self.has_sdS_xchg = False` and `self.xchg_aliases_dQaccum = False`: the compute warp issues a bulk DSMEM copy directly from `sdS_dK` into the peer CTA's `sdS`, removing both the extra SMEM buffer and the `dQaccum_empty_mbar` dependency.
4. Because the separate `sKt` tile for cluster-wide `dQ` reduction consumes `64 × 256 × 2B = 32 KiB` (`1 KiB` below the `227 KiB` SM100 SMEM ceiling), `hdim=256` sets `dQ_reduce_ncol = 16, sdQaccum_stage = 2` (`16 KiB` staging across two 16-column stages).

---

## 3. Unified `AccumTileOrder` and SMEM-Free Postprocess (`bwd_accum_tile.py`)

Commit `965c9212e6be33c821ee9f2b461564c5c91a2503` introduces `flash_attn/cute/bwd_accum_tile.py` (L1–84), unifying the FP32 accumulate tile layout between `FlashAttentionBackwardSm100` (`_check_accum_tile_orders`) and `FlashAttentionBackwardPostprocess`:

$$\begin{aligned}
\mathrm{idx}_{\text{ROW\_MAJOR}}(r, c) &= r \cdot d + c \\
\mathrm{idx}_{\text{TMEM\_FRAG\_1CTA}}(r, c) &= 4r + (c \bmod 4) + 4R \cdot \lfloor c / 4 \rfloor \\
\mathrm{idx}_{\text{TMEM\_FRAG\_2CTA}}(r, c) &= 4(r \bmod 64) + 64d \cdot \lfloor r / 64 \rfloor + (c' \bmod 4) + 512 \cdot \lfloor c' / 4 \rfloor + 256 \cdot \lfloor 2c / d \rfloor,\quad c' = c \bmod (d/2)
\end{aligned}$$

| Kernel Config `(use_2cta_instrs, tile_n)` | `acc_m64` | `dQaccum_order` | `dKVaccum_order` (GQA `dK/dV`) | Postprocess Global Memory Access |
|---|---|---|---|---|
| `(False, 128)` (1-CTA `M=128`) | `False` | `TMEM_FRAG_1CTA` | `TMEM_FRAG_1CTA` | Each thread loads `4` contiguous FP32s (`512 B` coalesced per warp) directly from GMEM to registers (**zero `cp.async` SMEM buffer**) |
| `(True, 128)` (2-CTA `M=128`) | `False` | `TMEM_FRAG_2CTA` | `TMEM_FRAG_1CTA` | Direct GMEM-to-register loads via `TMEM_FRAG_2CTA` (`dQ`) and `TMEM_FRAG_1CTA` (`dK/dV`) |
| **`(True, 64)` (2-CTA `M=64`, `hdim=256`)** | **`True`** | **`TMEM_FRAG_2CTA`** | **`ROW_MAJOR`** | `dK/dV` written in two `32 × 256` half-tiles as `ROW_MAJOR`; postprocess loads directly into registers |

---

## 4. Deterministic Verification (`experiments/fa4_sm100_hd256_bwd_m64_sim/run.py`)

Run `python experiments/fa4_sm100_hd256_bwd_m64_sim/run.py --check` to verify all 5 SM100 TMEM configurations and 3 `AccumTileOrder` bijections:
- `hd256_1cta_m128_overflow`: `tmem_end = 768 > 512` (`256`-column overflow);
- `hd256_2cta_m128_serial_overflow`: `tmem_end = 640 > 512` (`128`-column overflow);
- `hd256_2cta_m64_serial_valid`: `acc_m64 = True, lane_rows = 2`, `tmem_end = 512`, `has_sdS_xchg = False`, `skt_separate_bytes = 32768 B`;
- `AccumTileOrder` is bijective over all `32,768` coordinates of `(128, 256)` and matches all `400` 2-CTA `dQ` writer coordinates and `24` `M=64` half-tile `dK/dV` coordinates.

---

## 5. Primary Sources & Evidence Chain

| Source ID | Repository & Commit / File Path | Evidence Class |
|---|---|---|
| `flash-attention-sm100-hd256-m64-bwd-2026` | `Dao-AILab/flash-attention` commit `2b00d76f222c842b245fd98847223c898bc476db` (`flash_attn/cute/flash_bwd_sm100.py`, blob `9aedeeaadac1`) | `implementation` |
| `flash-attention-sm100-bwd-accum-tile-2026` | `Dao-AILab/flash-attention` commit `965c9212e6be33c821ee9f2b461564c5c91a2503` (`flash_attn/cute/bwd_accum_tile.py`, blob `9f75d17e248d`; `flash_bwd_postprocess.py`, blob `82dddba5df03`; `flash_bwd_sm100.py`, blob `67cb5b41707e`) | `implementation` |
| `flash-attention-sm100-bwd-interface-2026` | `Dao-AILab/flash-attention` commit `965c9212e6be33c821ee9f2b461564c5c91a2503` (`flash_attn/cute/interface.py`, blob `20636fa5d2e7`) | `implementation` |
