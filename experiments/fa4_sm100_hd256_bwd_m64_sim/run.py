#!/usr/bin/env python3
"""Deterministic simulator for FlashAttention-4 Blackwell SM100 hdim=256 2-CTA Backward:
M=64 TMEM accumulator layout, zero-extra-buffer DSMEM dS exchange, and AccumTileOrder mapping.

Models the exact arithmetic and layout invariants from Dao-AILab/flash-attention
commits 2b00d76f222c842b245fd98847223c898bc476db and 965c9212e6be33c821ee9f2b461564c5c91a2503:
- flash_attn/cute/flash_bwd_sm100.py:
  - M=128 vs M=64 (2-CTA 64-row KV tile) TMEM column budget on SM100 (512 columns max)
  - TWO_CTA_SERIAL schedule with tile_m=128, tile_n=64, acc_m64=True
  - Elimination of sdS_xchg buffer when P/dS^T reside in SMEM (sdS_dK)
  - dQ_reduce_ncol=16, sdQaccum_stage=2 (16 KiB staging + 32 KiB sKt within 227 KiB SMEM)
- flash_attn/cute/bwd_accum_tile.py:
  - Closed-form bijective index mappings for TMEM_FRAG_1CTA, TMEM_FRAG_2CTA, and ROW_MAJOR
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass


SM100_MAX_TMEM_COLS = 512
SM100_MAX_SMEM_BYTES = 227 * 1024  # 232,448 B


@dataclass(frozen=True)
class Sm100BwdTmemConfig:
    name: str
    tile_m: int
    tile_n: int
    tile_hdim: int
    tile_hdimv: int
    use_2cta_instrs: bool
    cta_schedule: str  # "DEFAULT" or "TWO_CTA_SERIAL"
    acc_m64: bool
    lane_rows: int
    tmem_cols_dV: int
    tmem_cols_dK: int
    tmem_cols_S: int
    tmem_cols_dP: int
    tmem_cols_dQ: int
    tmem_cols_P: int
    tmem_cols_dS: int
    tmem_dV_offset: int
    tmem_dK_offset: int
    tmem_S_offset: int
    tmem_dP_offset: int
    tmem_dQ_offset: int
    tmem_end: int
    fits_in_512_cols: bool
    has_sdS_xchg: bool
    dq_reduce_ncol: int
    sdqaccum_stage: int
    sdqaccum_bytes: int
    skt_separate_bytes: int
    dQaccum_order: str
    dKVaccum_order: str


def evaluate_sm100_bwd_tmem_config(
    name: str,
    tile_m: int,
    tile_n: int,
    tile_hdim: int,
    tile_hdimv: int,
    use_2cta_instrs: bool,
    cta_schedule: str,
) -> Sm100BwdTmemConfig:
    cta_group_size = 2 if use_2cta_instrs else 1
    acc_m64 = use_2cta_instrs and tile_n == 64
    lane_rows = 2 if acc_m64 else 1

    tmem_cols_S = tile_m // lane_rows
    tmem_cols_dP = tile_m // lane_rows
    tmem_cols_dV = tile_hdimv // lane_rows
    tmem_cols_dK = tile_hdim // lane_rows
    tmem_cols_dQ = tile_hdim // cta_group_size
    tmem_cols_P = 0 if acc_m64 else tmem_cols_S // 2
    tmem_cols_dS = 0 if acc_m64 else tmem_cols_dP // 2

    if cta_schedule == "TWO_CTA_SERIAL":
        tmem_dV_offset = 0
        tmem_dK_offset = tmem_dV_offset + tmem_cols_dV
        tmem_S_offset = tmem_dK_offset + tmem_cols_dK
        tmem_P_offset = tmem_S_offset
        tmem_dP_offset = SM100_MAX_TMEM_COLS - tmem_cols_dP
        tmem_dS_offset = tmem_dP_offset
        tmem_dQ_offset = SM100_MAX_TMEM_COLS - tmem_cols_dQ
    else:
        tmem_S_offset = 0
        tmem_dV_offset = tmem_S_offset + tmem_cols_S
        tmem_dP_offset = tmem_dV_offset + tmem_cols_dV
        tmem_dQ_offset = (
            tmem_S_offset + tmem_cols_S - tmem_cols_dQ
            if use_2cta_instrs
            else tmem_dP_offset
        )
        tmem_dK_offset = tmem_dP_offset + tmem_cols_dP
        tmem_P_offset = tmem_S_offset
        tmem_dS_offset = tmem_dP_offset

    tmem_end = max(
        tmem_S_offset + tmem_cols_S,
        tmem_dP_offset + tmem_cols_dP,
        tmem_dV_offset + tmem_cols_dV,
        tmem_dK_offset + tmem_cols_dK,
        tmem_dQ_offset + tmem_cols_dQ,
        tmem_P_offset + tmem_cols_P,
        tmem_dS_offset + tmem_cols_dS,
    )
    fits_in_512_cols = (
        tmem_end <= SM100_MAX_TMEM_COLS
        and tmem_dP_offset >= tmem_dK_offset + tmem_cols_dK
    )

    has_sdS_xchg = use_2cta_instrs and not acc_m64
    if cta_schedule == "TWO_CTA_SERIAL":
        if tile_hdim == 192:
            dq_reduce_ncol = 24
            sdqaccum_stage = 2
        else:
            dq_reduce_ncol = 16
            sdqaccum_stage = 2
    else:
        dq_reduce_ncol = 16 if use_2cta_instrs else 32
        sdqaccum_stage = 2

    # Each dQaccum stage holds (tile_m // cta_group_size) * dq_reduce_ncol * 4 bytes (or 128 lanes * ncol * 4 bytes / 2)
    # Specifically on SM100: c_smem_layout_staged has (128, dq_reduce_ncol // 2, sdqaccum_stage) in FP32 (4B)
    sdqaccum_bytes = 128 * (dq_reduce_ncol // 2) * sdqaccum_stage * 4 if use_2cta_instrs else 128 * dq_reduce_ncol * sdqaccum_stage * 4
    skt_separate_bytes = (tile_n * tile_hdim * 2) if (cta_schedule == "TWO_CTA_SERIAL" and tile_hdim == 256) else 0

    dQaccum_order = "TMEM_FRAG_2CTA" if use_2cta_instrs else "TMEM_FRAG_1CTA"
    dKVaccum_order = "ROW_MAJOR" if acc_m64 else "TMEM_FRAG_1CTA"

    return Sm100BwdTmemConfig(
        name=name,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_hdim=tile_hdim,
        tile_hdimv=tile_hdimv,
        use_2cta_instrs=use_2cta_instrs,
        cta_schedule=cta_schedule,
        acc_m64=acc_m64,
        lane_rows=lane_rows,
        tmem_cols_dV=tmem_cols_dV,
        tmem_cols_dK=tmem_cols_dK,
        tmem_cols_S=tmem_cols_S,
        tmem_cols_dP=tmem_cols_dP,
        tmem_cols_dQ=tmem_cols_dQ,
        tmem_cols_P=tmem_cols_P,
        tmem_cols_dS=tmem_cols_dS,
        tmem_dV_offset=tmem_dV_offset,
        tmem_dK_offset=tmem_dK_offset,
        tmem_S_offset=tmem_S_offset,
        tmem_dP_offset=tmem_dP_offset,
        tmem_dQ_offset=tmem_dQ_offset,
        tmem_end=tmem_end,
        fits_in_512_cols=fits_in_512_cols,
        has_sdS_xchg=has_sdS_xchg,
        dq_reduce_ncol=dq_reduce_ncol,
        sdqaccum_stage=sdqaccum_stage,
        sdqaccum_bytes=sdqaccum_bytes,
        skt_separate_bytes=skt_separate_bytes,
        dQaccum_order=dQaccum_order,
        dKVaccum_order=dKVaccum_order,
    )


def accum_tile_crd2idx(order: str, rows: int, hdim: int, row: int, col: int) -> int:
    """Exact index mapping from flash_attn/cute/bwd_accum_tile.py::accum_tile_layout."""
    assert 0 <= row < rows and 0 <= col < hdim
    assert hdim % 8 == 0
    if order == "ROW_MAJOR":
        return row * hdim + col
    if order == "TMEM_FRAG_1CTA":
        c0 = col % 4
        c1 = col // 4
        return row * 4 + c0 * 1 + c1 * (4 * rows)
    if order == "TMEM_FRAG_2CTA":
        assert rows == 128
        half_m = rows // 2  # 64
        r0 = row % half_m
        r1 = row // half_m
        half_col = col // (hdim // 2)
        local_col = col % (hdim // 2)
        c0 = local_col % 4
        c1 = local_col // 4
        return (
            r0 * 4
            + r1 * (half_m * hdim)
            + c0 * 1
            + c1 * (4 * half_m * 2)
            + half_col * (4 * half_m)
        )
    raise ValueError(f"Unknown AccumTileOrder: {order}")


def verify_accum_tile_bijection_and_writer_match(rows: int = 128, hdim: int = 256, ncol: int = 16) -> dict:
    """Verify bijection across all 3 AccumTileOrder layouts and match against SM100 1-D epilogue copies."""
    total_elems = rows * hdim
    bijection_ok = {}
    for order in ("ROW_MAJOR", "TMEM_FRAG_1CTA", "TMEM_FRAG_2CTA"):
        seen = {
            accum_tile_crd2idx(order, rows, hdim, r, c)
            for r in range(rows)
            for c in range(hdim)
        }
        bijection_ok[order] = len(seen) == total_elems and min(seen) == 0 and max(seen) == total_elems - 1

    # Check 2-CTA dQ reduce writer formula from _check_accum_tile_orders (lines 376-389)
    cta_group_size = 2
    dQaccum_reduce_stage = hdim // ncol
    stages_per_cta = dQaccum_reduce_stage // cta_group_size
    dq_2cta_matches = 0
    for r in range(cta_group_size):
        for s in range(stages_per_cta):
            for t in (0, 1, 63, 64, 127):
                for j in (0, 1, 3, 4, ncol - 1):
                    flat = (s + stages_per_cta * r) * 128 * ncol + 4 * t + (j % 4) + 512 * (j // 4)
                    row = (t % 64) + 64 * r
                    col = s * ncol + j + (hdim // 2) * (t // 64)
                    idx = accum_tile_crd2idx("TMEM_FRAG_2CTA", rows, hdim, row, col)
                    assert idx == flat, f"Mismatch at r={r}, s={s}, t={t}, j={j}: {idx} != {flat}"
                    dq_2cta_matches += 1

    # Check M=64 dK/dV half-tile block property (lines 390-402)
    tile_n = 64
    rows_per_half = tile_n // 2
    dkv_m64_half_matches = 0
    for half in range(2):
        for r in (0, 1, rows_per_half - 1):
            for c in (0, 1, 4, hdim - 1):
                full_idx = accum_tile_crd2idx("ROW_MAJOR", tile_n, hdim, half * rows_per_half + r, c)
                half_idx = half * rows_per_half * hdim + accum_tile_crd2idx("ROW_MAJOR", rows_per_half, hdim, r, c)
                assert full_idx == half_idx
                dkv_m64_half_matches += 1

    # Postprocess warp contiguous read check: 32 threads * 4 FP32 = 128 floats = 512 bytes contiguous
    warp_contiguous_bytes = 32 * 4 * 4
    return {
        "total_elements": total_elems,
        "bijection_verified": bijection_ok,
        "dq_2cta_writer_sample_matches": dq_2cta_matches,
        "dkv_m64_half_tile_matches": dkv_m64_half_matches,
        "postprocess_warp_read_bytes": warp_contiguous_bytes,
    }


def build_summary() -> dict:
    configs = [
        evaluate_sm100_bwd_tmem_config(
            name="hd128_1cta_m128_default",
            tile_m=128,
            tile_n=128,
            tile_hdim=128,
            tile_hdimv=128,
            use_2cta_instrs=False,
            cta_schedule="DEFAULT",
        ),
        evaluate_sm100_bwd_tmem_config(
            name="hd128_2cta_m128_default",
            tile_m=128,
            tile_n=128,
            tile_hdim=128,
            tile_hdimv=128,
            use_2cta_instrs=True,
            cta_schedule="DEFAULT",
        ),
        evaluate_sm100_bwd_tmem_config(
            name="hd256_1cta_m128_overflow",
            tile_m=128,
            tile_n=128,
            tile_hdim=256,
            tile_hdimv=256,
            use_2cta_instrs=False,
            cta_schedule="DEFAULT",
        ),
        evaluate_sm100_bwd_tmem_config(
            name="hd256_2cta_m128_serial_overflow",
            tile_m=128,
            tile_n=128,
            tile_hdim=256,
            tile_hdimv=256,
            use_2cta_instrs=True,
            cta_schedule="TWO_CTA_SERIAL",
        ),
        evaluate_sm100_bwd_tmem_config(
            name="hd256_2cta_m64_serial_valid",
            tile_m=128,
            tile_n=64,
            tile_hdim=256,
            tile_hdimv=256,
            use_2cta_instrs=True,
            cta_schedule="TWO_CTA_SERIAL",
        ),
    ]
    layout_verification = verify_accum_tile_bijection_and_writer_match(rows=128, hdim=256, ncol=16)
    by_name = {c.name: asdict(c) for c in configs}
    return {
        "sm100_max_tmem_cols": SM100_MAX_TMEM_COLS,
        "sm100_max_smem_bytes": SM100_MAX_SMEM_BYTES,
        "configs": by_name,
        "layout_verification": layout_verification,
    }


def verify_summary(summary: dict) -> None:
    cfgs = summary["configs"]
    # 1. hd256 with M=128 1-CTA overflows 512 TMEM columns (needs 768 cols)
    c_1cta_256 = cfgs["hd256_1cta_m128_overflow"]
    assert c_1cta_256["tmem_end"] == 768
    assert c_1cta_256["fits_in_512_cols"] is False

    # 2. hd256 with M=128 2-CTA serial still overflows (dV=256 + dK=256 = 512, leaving 0 non-overlapping cols for dP=128)
    c_2cta_m128_256 = cfgs["hd256_2cta_m128_serial_overflow"]
    assert c_2cta_m128_256["tmem_S_offset"] == 512
    assert c_2cta_m128_256["tmem_end"] == 640
    assert c_2cta_m128_256["fits_in_512_cols"] is False

    # 3. hd256 with M=64 (tile_n=64) 2-CTA serial fits cleanly inside 512 TMEM columns!
    c_m64 = cfgs["hd256_2cta_m64_serial_valid"]
    assert c_m64["acc_m64"] is True
    assert c_m64["lane_rows"] == 2
    assert c_m64["tmem_cols_dV"] == 128
    assert c_m64["tmem_cols_dK"] == 128
    assert c_m64["tmem_cols_S"] == 64
    assert c_m64["tmem_cols_dP"] == 64
    assert c_m64["tmem_cols_dQ"] == 128
    assert c_m64["tmem_cols_P"] == 0 and c_m64["tmem_cols_dS"] == 0
    assert c_m64["tmem_dV_offset"] == 0
    assert c_m64["tmem_dK_offset"] == 128
    assert c_m64["tmem_S_offset"] == 256
    assert c_m64["tmem_dP_offset"] == 448
    assert c_m64["tmem_dQ_offset"] == 384
    assert c_m64["tmem_end"] == 512
    assert c_m64["fits_in_512_cols"] is True
    assert c_m64["has_sdS_xchg"] is False
    assert c_m64["dq_reduce_ncol"] == 16 and c_m64["sdqaccum_stage"] == 2
    assert c_m64["sdqaccum_bytes"] == 8192 * 1  # per CTA (16 KiB across the 2 stages of 128*8*4B)
    assert c_m64["skt_separate_bytes"] == 32768
    assert c_m64["dQaccum_order"] == "TMEM_FRAG_2CTA"
    assert c_m64["dKVaccum_order"] == "ROW_MAJOR"

    # 4. Layout bijections and writer sample checks
    lv = summary["layout_verification"]
    assert all(lv["bijection_verified"].values())
    assert lv["dq_2cta_writer_sample_matches"] == 400
    assert lv["dkv_m64_half_tile_matches"] == 24
    assert lv["postprocess_warp_read_bytes"] == 512


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify deterministic invariants")
    parser.add_argument("--json", action="store_true", help="Print JSON output")
    args = parser.parse_args()

    summary = build_summary()
    if args.check:
        verify_summary(summary)
    if args.json or not args.check:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print("fa4_sm100_hd256_bwd_m64_sim: CHECK PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
