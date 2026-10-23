#!/usr/bin/env python3
"""Phase 64 Track A deterministic offline simulation and verification guard:
1. FlashInfer Cake KDA Context-Parallel (KCP) saved-context backward carry plane reuse (#7117, ddc9524)
2. FlashInfer Cake QSA NVFP4 sparse paged GQA decode attention (load K first, evict_first, 1-prmt dequant) (#7103, 34d28b3)
3. FlashInfer Cake MegaMHC SM107 evict-first residual stream & _claim_launch_epochs grid-sharing guard (#7113, 85e886c)
4. TileLang VectorizeLoop non-unit step rejection (#3345, cb26ee0) & CUDA AtomicStore/AtomicMax destination type preservation (#3307, 2573d47)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

SUMMARY_PATH = Path(__file__).with_name("summary.json")


def simulate_kcp_carry_plane_reuse() -> dict[str, Any]:
    """Model FlashInfer PR #7117 (ddc9524): KCP saved-context backward carry plane reuse on GB300 (sm_103a)."""
    # Selected representative rows from the 144-row GB300 (sm_103a, H=64) table in PR #7117
    b3_rows = [
        {"pack": "Q1 (249,856 tok / 4 docs)", "cp": 2, "layout": "contiguous", "prev_s1_ms": 66.871, "prev_s2_ms": 66.863, "curr_s1_ms": 65.338, "curr_s2_ms": 65.288, "speedup_s1": 1.0235, "speedup_s2": 1.0241},
        {"pack": "Q2 (226,336 tok / 2 docs)", "cp": 2, "layout": "zigzag", "prev_s1_ms": 63.115, "prev_s2_ms": 63.099, "curr_s1_ms": 61.520, "curr_s2_ms": 61.515, "speedup_s1": 1.0259, "speedup_s2": 1.0257},
        {"pack": "Q1 (249,856 tok / 4 docs)", "cp": 4, "layout": "contiguous", "prev_s1_ms": 34.272, "prev_s2_ms": 34.210, "curr_s1_ms": 33.375, "curr_s2_ms": 33.349, "speedup_s1": 1.0269, "speedup_s2": 1.0258},
        {"pack": "Q4 (240,160 tok / 3 docs)", "cp": 4, "layout": "zigzag", "prev_s1_ms": 33.841, "prev_s2_ms": 33.803, "curr_s1_ms": 32.941, "curr_s2_ms": 32.941, "speedup_s1": 1.0273, "speedup_s2": 1.0262},
        {"pack": "Q1 (249,856 tok / 4 docs)", "cp": 8, "layout": "contiguous", "prev_s1_ms": 18.453, "prev_s2_ms": 18.458, "curr_s1_ms": 17.865, "curr_s2_ms": 17.847, "speedup_s1": 1.0329, "speedup_s2": 1.0342},
        {"pack": "Q4 (240,160 tok / 3 docs)", "cp": 8, "layout": "contiguous", "prev_s1_ms": 17.851, "prev_s2_ms": 17.860, "curr_s1_ms": 17.278, "curr_s2_ms": 17.251, "speedup_s1": 1.0332, "speedup_s2": 1.0353},
        {"pack": "Q1 (249,856 tok / 4 docs)", "cp": 16, "layout": "contiguous", "prev_s1_ms": 10.116, "prev_s2_ms": 10.097, "curr_s1_ms": 9.762, "curr_s2_ms": 9.767, "speedup_s1": 1.0363, "speedup_s2": 1.0338},
        {"pack": "Q4 (240,160 tok / 3 docs)", "cp": 16, "layout": "contiguous", "prev_s1_ms": 9.794, "prev_s2_ms": 9.788, "curr_s1_ms": 9.443, "curr_s2_ms": 9.423, "speedup_s1": 1.0372, "speedup_s2": 1.0387},
    ]

    # Exact saved carry plane byte formula from test_cake_kda_train_kcp.py:
    # plane_slots = n_frag * (P - 1) + max(n_in, n_out)
    # plane_bytes = plane_slots * H * STATE_TILE_BYTES if P > 1 else 0
    # where H = 64, STATE_TILE_BYTES = 128 * 128 * 4 = 65536 bytes (64 KiB per head state tile, 4 MiB across 64 heads)
    state_tile_bytes = 65536
    heads = 64
    bytes_per_slot_all_heads = heads * state_tile_bytes  # 4,194,304 bytes = 4.0 MiB

    def kcp_carry_plane_bytes(n_frag: int, p_segments: int, n_in: int, n_out: int) -> int:
        if p_segments <= 1:
            return 0
        plane_slots = n_frag * (p_segments - 1) + max(n_in, n_out)
        return plane_slots * bytes_per_slot_all_heads

    # Unit test case from test_byte_queries_and_knobs: n_frag=1, P=3, max(n_in, n_out)=1 -> plane_slots = 1*2 + 1 = 3 -> 12 MiB
    min_plane_bytes = kcp_carry_plane_bytes(n_frag=1, p_segments=3, n_in=1, n_out=1)
    max_plane_bytes = kcp_carry_plane_bytes(n_frag=6, p_segments=4, n_in=2, n_out=2)
    single_seg_bytes = kcp_carry_plane_bytes(n_frag=4, p_segments=1, n_in=2, n_out=2)

    geomean_s1 = math.exp(sum(math.log(r["speedup_s1"]) for r in b3_rows) / len(b3_rows))
    geomean_s2 = math.exp(sum(math.log(r["speedup_s2"]) for r in b3_rows) / len(b3_rows))

    return {
        "hardware": "NVIDIA GB300 (sm_103a), CUDA 13.0, driver 580.159.03, H=64",
        "total_matrix_rows": 144,
        "rows_faster_gt_2pct_both_sessions": 34,
        "rows_within_2pct_both_sessions": 101,
        "rows_slower_gt_2pct_both_sessions": 0,
        "b3_speedup_range_s1": [0.9990, 1.0372],
        "b3_speedup_range_s2": [0.9950, 1.0387],
        "sampled_b3_geomean_speedup_s1": round(geomean_s1, 4),
        "sampled_b3_geomean_speedup_s2": round(geomean_s2, 4),
        "carry_plane_min_mib": round(min_plane_bytes / (1024 * 1024), 2),
        "carry_plane_max_mib": round(max_plane_bytes / (1024 * 1024), 2),
        "p1_carry_plane_bytes": single_seg_bytes,
        "b3_rows": b3_rows,
    }


def simulate_qsa_nvfp4_decode() -> dict[str, Any]:
    """Model FlashInfer PR #7103 (34d28b3): Cake QSA NVFP4 sparse paged GQA decode attention on H100 SXM."""
    rows = [
        {"layout": "NHD", "batch_rows": 1, "prev_us": 9.98, "curr_us": 9.90, "speedup": 1.008},
        {"layout": "NHD", "batch_rows": 2, "prev_us": 11.26, "curr_us": 11.17, "speedup": 1.009},
        {"layout": "NHD", "batch_rows": 4, "prev_us": 12.82, "curr_us": 12.74, "speedup": 1.006},
        {"layout": "NHD", "batch_rows": 8, "prev_us": 16.66, "curr_us": 16.51, "speedup": 1.009},
        {"layout": "NHD", "batch_rows": 16, "prev_us": 23.65, "curr_us": 22.02, "speedup": 1.074},
        {"layout": "NHD", "batch_rows": 32, "prev_us": 36.42, "curr_us": 32.32, "speedup": 1.127},
        {"layout": "NHD", "batch_rows": 64, "prev_us": 58.59, "curr_us": 56.30, "speedup": 1.041},
        {"layout": "NHD", "batch_rows": 128, "prev_us": 99.81, "curr_us": 100.33, "speedup": 0.995},
        {"layout": "HND", "batch_rows": 1, "prev_us": 9.73, "curr_us": 9.74, "speedup": 0.998},
        {"layout": "HND", "batch_rows": 2, "prev_us": 11.07, "curr_us": 10.86, "speedup": 1.019},
        {"layout": "HND", "batch_rows": 4, "prev_us": 12.54, "curr_us": 12.42, "speedup": 1.010},
        {"layout": "HND", "batch_rows": 8, "prev_us": 17.39, "curr_us": 16.19, "speedup": 1.074},
        {"layout": "HND", "batch_rows": 16, "prev_us": 25.52, "curr_us": 23.01, "speedup": 1.109},
        {"layout": "HND", "batch_rows": 32, "prev_us": 38.83, "curr_us": 34.45, "speedup": 1.127},
        {"layout": "HND", "batch_rows": 64, "prev_us": 62.03, "curr_us": 59.82, "speedup": 1.037},
        {"layout": "HND", "batch_rows": 128, "prev_us": 103.15, "curr_us": 103.50, "speedup": 0.997},
    ]

    mid_batch = [r for r in rows if r["batch_rows"] in (16, 32, 64)]
    mid_batch_geomean = math.exp(sum(math.log(r["speedup"]) for r in mid_batch) / len(mid_batch))
    overall_geomean = math.exp(sum(math.log(r["speedup"]) for r in rows) / len(rows))
    peak_speedup = max(r["speedup"] for r in rows)

    return {
        "hardware": "NVIDIA H100 80GB HBM3 (SXM, 132 SMs), sm_90a, CUDA 13.4",
        "shape": "24 Q heads, 2 KV heads, head_dim=256, top_k=2051, page_size=16",
        "total_shapes": len(rows),
        "peak_speedup": peak_speedup,
        "mid_batch_16_32_64_geomean_speedup": round(mid_batch_geomean, 4),
        "overall_16_shape_geomean_speedup": round(overall_geomean, 4),
        "rows": rows,
    }


def simulate_megamhc_and_tilelang_guards() -> dict[str, Any]:
    """Verify MegaMHC _claim_launch_epochs grid-sharing guard (#7113) and TileLang VectorizeLoop/AtomicStore guards (#3345, #3307)."""

    # 1. MegaMHC _claim_launch_epochs storage & grid-sharing guard
    def validate_megamhc_epoch_claim(
        storage_offset: int,
        storage_nbytes: int,
        tensor_nbytes: int,
        active_grid: int | None,
        requested_grid: int,
    ) -> tuple[bool, str]:
        if storage_offset != 0 or storage_nbytes != tensor_nbytes:
            return False, "sliced_or_offset_view_refused"
        if active_grid is not None and active_grid != requested_grid:
            return False, f"mismatched_grid_refused:{active_grid}_vs_{requested_grid}"
        return True, "accepted"

    ok_same, status_same = validate_megamhc_epoch_claim(0, 1184, 1184, 64, 64)
    ok_diff, status_diff = validate_megamhc_epoch_claim(0, 1184, 1184, 64, 148)
    ok_slice, status_slice = validate_megamhc_epoch_claim(8, 1184, 592, None, 64)

    # 2. MegaMHC SM107 L2 cache policy selector: (num_tokens + 63) // 64 <= 64
    def select_mhc_io_policy(num_tokens: int) -> int:
        tile_rows = (num_tokens + 63) // 64
        return 1364590687093260288 if tile_rows <= 64 else 1152921504606846976

    policy_small = select_mhc_io_policy(4096)  # 64 tiles -> 0x12f0000000000000
    policy_large = select_mhc_io_policy(4097)  # 65 tiles -> 0x1000000000000000

    # 3. TileLang VectorizeLoop step guard (#3345)
    def verify_vectorize_loop_step(min_val: int, step_val: int | None) -> tuple[bool, str]:
        if min_val != 0:
            return False, "non_zero_min_rejected"
        if step_val is not None and step_val != 1:
            return False, f"non_unit_step_rejected:{step_val}"
        return True, "vectorized"

    vec_unit_ok, _ = verify_vectorize_loop_step(0, 1)
    vec_none_ok, _ = verify_vectorize_loop_step(0, None)
    vec_step2_ok, vec_step2_reason = verify_vectorize_loop_step(0, 2)

    # 4. TileLang CUDA AtomicMax signed int64 & AtomicStore destination type preservation (#3307)
    def simulate_signed_int64_atomic_max(cur_val: int, new_val: int, preserve_signed_ref: bool) -> int:
        if preserve_signed_ref:
            return max(cur_val, new_val)
        # Buggy path cast signed int64 to unsigned uint64 via cuda_cast<NT1>, making negative numbers huge positive
        u_cur = cur_val & 0xFFFFFFFFFFFFFFFF
        u_new = new_val & 0xFFFFFFFFFFFFFFFF
        u_max = max(u_cur, u_new)
        return u_max - (1 << 64) if (u_max & (1 << 63)) else u_max

    fixed_max = simulate_signed_int64_atomic_max(-5, 3, preserve_signed_ref=True)
    buggy_max = simulate_signed_int64_atomic_max(-5, 3, preserve_signed_ref=False)

    return {
        "megamhc_same_grid_accepted": ok_same,
        "megamhc_mismatched_grid_rejected": not ok_diff,
        "megamhc_mismatched_grid_reason": status_diff,
        "megamhc_sliced_view_rejected": not ok_slice,
        "megamhc_io_policy_le_4096": hex(policy_small),
        "megamhc_io_policy_gt_4096": hex(policy_large),
        "tilelang_vectorize_unit_step_accepted": vec_unit_ok and vec_none_ok,
        "tilelang_vectorize_step2_rejected": not vec_step2_ok,
        "tilelang_vectorize_step2_reason": vec_step2_reason,
        "tilelang_signed_int64_atomic_max_fixed": fixed_max,
        "tilelang_signed_int64_atomic_max_buggy": buggy_max,
    }


def build_summary() -> dict[str, Any]:
    kcp = simulate_kcp_carry_plane_reuse()
    qsa = simulate_qsa_nvfp4_decode()
    guards = simulate_megamhc_and_tilelang_guards()
    return {
        "experiment": "flashinfer-kcp-carry-qsa-nvfp4-mhc-tilelang",
        "phase": 64,
        "priority": 73,
        "provenance": "own-offline deterministic guard + upstream author-benchmark tables (FlashInfer #7117, #7103, #7113; TileLang #3345, #3307)",
        "kcp_carry_plane": kcp,
        "qsa_nvfp4_decode": qsa,
        "megamhc_and_tilelang_guards": guards,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Verify summary.json matches deterministic outputs")
    args = parser.parse_args()

    summary = build_summary()
    rendered = json.dumps(summary, indent=2, ensure_ascii=False) + "\n"

    if args.check:
        if not SUMMARY_PATH.exists():
            raise SystemExit(f"Missing {SUMMARY_PATH}")
        existing = SUMMARY_PATH.read_text(encoding="utf-8")
        if existing != rendered:
            raise SystemExit("summary.json is out of sync with deterministic run.py output")
        print("own-offline outputs verified for flashinfer-kcp-carry-qsa-nvfp4-mhc-tilelang")
        return 0

    SUMMARY_PATH.write_text(rendered, encoding="utf-8")
    print(f"Wrote {SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
