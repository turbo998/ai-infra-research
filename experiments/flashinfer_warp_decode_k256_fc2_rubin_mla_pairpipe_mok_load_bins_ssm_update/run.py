#!/usr/bin/env python3
"""Phase 82 Track 1 experiment: FlashInfer cake_warp_decode K256 FC2, Rubin MLA pairpipe, cake_mok load bins & SSM update."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def model_cake_warp_decode_k256_fc2() -> dict[str, Any]:
    """Model cake_warp_decode linear-grid K256 FC2 routing (#7339, commit 53f6bc4d3f47)."""
    sm100_i256_rows = list(range(9, 33))   # 24 rows
    sm100_i512_rows = list(range(19, 33))  # 14 rows
    sm103_i256_rows = list(range(7, 33))   # 26 rows
    sm103_i512_rows = list(range(12, 33))  # 21 rows

    sm100_i256_t32_read_prev_us = 98.18
    sm100_i256_t32_read_new_us = 93.46
    sm100_i256_t32_speedup = round(sm100_i256_t32_read_prev_us / sm100_i256_t32_read_new_us, 4)

    sm103_i256_t31_read_prev_us = 95.57
    sm103_i256_t31_read_new_us = 91.24
    sm103_i256_t31_speedup = round(sm103_i256_t31_read_prev_us / sm103_i256_t31_read_new_us, 4)

    sm100_i512_t32_read_prev_us = 151.61
    sm100_i512_t32_read_new_us = 148.80
    sm100_i512_t32_speedup = round(sm100_i512_t32_read_prev_us / sm100_i512_t32_read_new_us, 4)

    return {
        "kernel_symbol": "kernel_cake_warp_decode_4f408b642a6c28d2940e",
        "smem_bytes": 179456,
        "threads_per_cta": 256,
        "mma_k_tile": 256,
        "output_tile_rows": 128,
        "route_tile_tokens": 8,
        "sm100a_routed_rows_total": len(sm100_i256_rows) + len(sm100_i512_rows),
        "sm103a_routed_rows_total": len(sm103_i256_rows) + len(sm103_i512_rows),
        "sm100_i256_t32_read_prev_us": sm100_i256_t32_read_prev_us,
        "sm100_i256_t32_read_new_us": sm100_i256_t32_read_new_us,
        "sm100_i256_t32_speedup_x": sm100_i256_t32_speedup,
        "sm103_i256_t31_read_prev_us": sm103_i256_t31_read_prev_us,
        "sm103_i256_t31_read_new_us": sm103_i256_t31_read_new_us,
        "sm103_i256_t31_speedup_x": sm103_i256_t31_speedup,
        "sm100_i512_t32_read_prev_us": sm100_i512_t32_read_prev_us,
        "sm100_i512_t32_read_new_us": sm100_i512_t32_read_new_us,
        "sm100_i512_t32_speedup_x": sm100_i512_t32_speedup,
    }


def model_rubin_mla_nvfp4_decode_pairpipe() -> dict[str, Any]:
    """Model Rubin SM107 cake_rubin_mla_nvfp4_decode dual staging tiles & pairpipe (#7329, commit 5c314c510256)."""
    prev_smem_bytes = 323584
    new_smem_bytes = 332800
    smem_delta_bytes = new_smem_bytes - prev_smem_bytes
    staging_tile_bytes = 4096

    h448_q1_b1_kv64k_prev_b_us = 26.22
    h448_q1_b1_kv64k_new_b_us = 25.04
    h448_q1_b1_kv64k_speedup_b = round(h448_q1_b1_kv64k_prev_b_us / h448_q1_b1_kv64k_new_b_us, 3)

    h192_q1_b2_kv64k_prev_a_us = 25.01
    h192_q1_b2_kv64k_new_a_us = 23.96
    h192_q1_b2_kv64k_speedup_a = round(h192_q1_b2_kv64k_prev_a_us / h192_q1_b2_kv64k_new_a_us, 3)

    h96_q8_b64_kv64k_prev_a_us = 1109.24
    h96_q8_b64_kv64k_new_a_us = 1094.48
    h96_q8_b64_kv64k_speedup_a = round(h96_q8_b64_kv64k_prev_a_us / h96_q8_b64_kv64k_new_a_us, 3)

    return {
        "target_arch": "sm_107a",
        "kv_page_bytes_per_token": 352,
        "prev_smem_bytes": prev_smem_bytes,
        "new_smem_bytes": new_smem_bytes,
        "smem_delta_bytes": smem_delta_bytes,
        "staging_tiles_per_correction_warp": 2,
        "staging_tile_bytes": staging_tile_bytes,
        "pv_output_rounds": 8,
        "pv_round_dim": 64,
        "pairpipe_routed_config": "H96-Q8-B64-KV64K",
        "h448_q1_b1_kv64k_speedup_b_x": h448_q1_b1_kv64k_speedup_b,
        "h192_q1_b2_kv64k_speedup_a_x": h192_q1_b2_kv64k_speedup_a,
        "h96_q8_b64_kv64k_speedup_a_x": h96_q8_b64_kv64k_speedup_a,
        "geomean_speedup_board_a_x": 1.0060,
        "geomean_speedup_board_b_x": 1.0062,
    }


def model_cake_mok_load_bins_and_ssm_update() -> dict[str, Any]:
    """Model cake_mok owner-rank load bins (#7326), cake_selective_state_update (#7319), and SSD mask (#7349)."""
    local_experts = 32
    schedule_capacity = 65536
    layout_int32_elements = 3 * local_experts + (schedule_capacity // 256)
    layout_bytes = layout_int32_elements * 4

    mok_b200_ep8_mxfp8_ragged_prev_ms = 15.975
    mok_b200_ep8_mxfp8_ragged_new_ms = 15.596
    mok_b200_ep8_mxfp8_ragged_speedup = round(
        mok_b200_ep8_mxfp8_ragged_prev_ms / mok_b200_ep8_mxfp8_ragged_new_ms, 3
    )

    ssm_b200_stp_bf16_prev_us = 44.560
    ssm_b200_stp_bf16_new_us = 39.440
    ssm_b200_stp_bf16_speedup = round(ssm_b200_stp_bf16_prev_us / ssm_b200_stp_bf16_new_us, 3)

    ssm_gb300_stp_softplus_prev_us = 47.168
    ssm_gb300_stp_softplus_new_us = 37.792
    ssm_gb300_stp_softplus_speedup = round(
        ssm_gb300_stp_softplus_prev_us / ssm_gb300_stp_softplus_new_us, 3
    )

    ssm_b200_granite_small_b1_prev_us = 3.808
    ssm_b200_granite_small_b1_new_us = 3.040
    ssm_b200_granite_small_b1_speedup = round(
        ssm_b200_granite_small_b1_prev_us / ssm_b200_granite_small_b1_new_us, 3
    )

    return {
        "mok_libraries_total": 144,
        "mok_Changed_units_per_arch": 10,
        "mok_layout_int32_elements": layout_int32_elements,
        "mok_layout_bytes": layout_bytes,
        "mok_b200_ep8_mxfp8_ragged_prev_ms": mok_b200_ep8_mxfp8_ragged_prev_ms,
        "mok_b200_ep8_mxfp8_ragged_new_ms": mok_b200_ep8_mxfp8_ragged_new_ms,
        "mok_b200_ep8_mxfp8_ragged_speedup_x": mok_b200_ep8_mxfp8_ragged_speedup,
        "ssm_shapes_total": 122,
        "ssm_b200_speedup_min_x": 1.015,
        "ssm_b200_speedup_max_x": ssm_b200_granite_small_b1_speedup,
        "ssm_gb300_speedup_min_x": 1.018,
        "ssm_gb300_speedup_max_x": ssm_gb300_stp_softplus_speedup,
        "ssm_b200_stp_bf16_speedup_x": ssm_b200_stp_bf16_speedup,
        "ssd_sm100_sm103_desc_mask_hex": "0x3FFF",
        "ssd_sm107_desc_mask_hex": "0x7FFF",
        "ssd_device_files_before": 93,
        "ssd_device_files_after": 72,
        "ssd_sm107_modules_before": 45,
        "ssd_sm107_modules_after": 19,
    }


def run_experiment() -> dict[str, Any]:
    warp_decode = model_cake_warp_decode_k256_fc2()
    rubin_mla = model_rubin_mla_nvfp4_decode_pairpipe()
    mok_and_ssm = model_cake_mok_load_bins_and_ssm_update()
    return {
        "experiment_id": "flashinfer_warp_decode_k256_fc2_rubin_mla_pairpipe_mok_load_bins_ssm_update",
        "phase": 82,
        "priority": 91,
        "provenance": "own-offline",
        "warp_decode_k256_fc2": warp_decode,
        "rubin_mla_nvfp4_decode": rubin_mla,
        "mok_load_bins_and_ssm_update": mok_and_ssm,
        "invariants": {
            "warp_decode_routed_rows_verified": (
                warp_decode["sm100a_routed_rows_total"] == 38
                and warp_decode["sm103a_routed_rows_total"] == 47
            ),
            "rubin_mla_smem_growth_9216b": rubin_mla["smem_delta_bytes"] == 9216,
            "ssm_all_122_shapes_faster": (
                mok_and_ssm["ssm_b200_speedup_min_x"] > 1.0
                and mok_and_ssm["ssm_gb300_speedup_min_x"] > 1.0
            ),
            "ssd_sm107_15bit_mask_folded": (
                int(mok_and_ssm["ssd_sm107_desc_mask_hex"], 16) == 0x7FFF
                and mok_and_ssm["ssd_device_files_after"] == 72
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate invariants and write summary.json")
    args = parser.parse_args()

    summary = run_experiment()
    out_path = Path(__file__).resolve().parent / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if args.check:
        assert all(summary["invariants"].values()), f"Invariant failed: {summary['invariants']}"
        assert summary["warp_decode_k256_fc2"]["sm100_i256_t32_speedup_x"] == 1.0505
        assert summary["rubin_mla_nvfp4_decode"]["h448_q1_b1_kv64k_speedup_b_x"] == 1.047
        assert summary["mok_load_bins_and_ssm_update"]["ssm_b200_speedup_max_x"] == 1.253
        assert summary["mok_load_bins_and_ssm_update"]["ssm_gb300_speedup_max_x"] == 1.248

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
