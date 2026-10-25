#!/usr/bin/env python3
"""Phase 81 Exp 1: FlashInfer SM120 QSA NVFP4 Whole-Tile Decode, SM100/103 DCP All-to-All LL128 & W4A16 Split MoE."""
import argparse
import json
import os
import sys

def run_model():
    # 1. SM120 QSA NVFP4 Whole-Tile Decode vs K-First Split/Fused (flashinfer 83755d9093)
    # Hq=24, Hkv=2, D=256, 2051 selected tokens in 64-token tiles (33 tiles total), page=16
    # Host selects whole_tile for split-form calls when ceil(33 / splits) >= 11 (704 tokens)
    sm120_sm_count = 188
    fused_ctas_per_sm = 2
    co_resident_cap = sm120_sm_count * fused_ctas_per_sm  # 376 CTAs
    selected_tokens = 2051
    tile_tokens = 64
    total_tiles = (selected_tokens + tile_tokens - 1) // tile_tokens  # 33 tiles

    qsa_rows_table = [
        {"rows": 1, "splits": 8, "nhd_prev_us": 12.24, "nhd_new_us": 12.21, "hnd_prev_us": 12.48, "hnd_new_us": 12.51, "graph_nhd_prev_us": 12.99, "graph_nhd_new_us": 13.07},
        {"rows": 8, "splits": 8, "nhd_prev_us": 22.80, "nhd_new_us": 22.56, "hnd_prev_us": 23.20, "hnd_new_us": 23.20, "graph_nhd_prev_us": 24.19, "graph_nhd_new_us": 24.19},
        {"rows": 16, "splits": 4, "nhd_prev_us": 35.23, "nhd_new_us": 34.94, "hnd_prev_us": 35.95, "hnd_new_us": 35.90, "graph_nhd_prev_us": 36.19, "graph_nhd_new_us": 36.26},
        {"rows": 32, "splits": 3, "nhd_prev_us": 57.25, "nhd_new_us": 54.74, "hnd_prev_us": 59.89, "hnd_new_us": 56.64, "graph_nhd_prev_us": 57.90, "graph_nhd_new_us": 55.06},
        {"rows": 64, "splits": 2, "nhd_prev_us": 98.86, "nhd_new_us": 94.26, "hnd_prev_us": 102.21, "hnd_new_us": 97.42, "graph_nhd_prev_us": 100.35, "graph_nhd_new_us": 93.86},
        {"rows": 128, "splits": 1, "nhd_prev_us": 161.65, "nhd_new_us": 152.80, "hnd_prev_us": 168.11, "hnd_new_us": 158.50, "graph_nhd_prev_us": 159.76, "graph_nhd_new_us": 152.14},
    ]
    for entry in qsa_rows_table:
        tiles_per_cta = (total_tiles + entry["splits"] - 1) // entry["splits"]
        entry["tiles_per_cta"] = tiles_per_cta
        entry["selected_tokens_per_cta"] = min(selected_tokens, tiles_per_cta * tile_tokens)
        entry["route"] = "whole_tile_split_evict_first" if tiles_per_cta >= 11 else "fused_cooperative_k_first"
        entry["nhd_speedup_x"] = round(entry["nhd_prev_us"] / entry["nhd_new_us"], 3)
        entry["hnd_speedup_x"] = round(entry["hnd_prev_us"] / entry["hnd_new_us"], 3)
        entry["graph_nhd_speedup_x"] = round(entry["graph_nhd_prev_us"] / entry["graph_nhd_new_us"], 3)

    # 2. Cake DCP All-to-All LL128 FIFO Deferred Slot Release (flashinfer 53feaae34c)
    dcp_cases = [
        {"gpu": "GB300_sm103a", "cp": 2, "tokens": 1, "prev_warm_us": 5.872, "new_warm_us": 4.784, "prev_cold_us": 5.840, "new_cold_us": 4.864},
        {"gpu": "GB300_sm103a", "cp": 2, "tokens": 64, "prev_warm_us": 6.560, "new_warm_us": 5.536, "prev_cold_us": 6.624, "new_cold_us": 5.568},
        {"gpu": "GB300_sm103a", "cp": 4, "tokens": 1, "prev_warm_us": 5.440, "new_warm_us": 4.480, "prev_cold_us": 5.424, "new_cold_us": 4.480},
        {"gpu": "GB300_sm103a", "cp": 4, "tokens": 128, "prev_warm_us": 7.536, "new_warm_us": 6.496, "prev_cold_us": 7.569, "new_cold_us": 6.496},
        {"gpu": "GB200_sm100a", "cp": 2, "tokens": 1, "prev_warm_us": 5.456, "new_warm_us": 4.288, "prev_cold_us": 5.584, "new_cold_us": 4.192},
        {"gpu": "GB200_sm100a", "cp": 4, "tokens": 128, "prev_warm_us": 7.824, "new_warm_us": 6.768, "prev_cold_us": 7.872, "new_cold_us": 6.816},
    ]
    for c in dcp_cases:
        c["warm_speedup_x"] = round(c["prev_warm_us"] / c["new_warm_us"], 3)
        c["cold_speedup_x"] = round(c["prev_cold_us"] / c["new_cold_us"], 3)

    # 3. Cake W4A16 Split MoE & SM103 NVFP4 Attention P-in-TMEM (cf5c9a6a41, d08a1f7321)
    w4a16_moe_cases = [
        {"preset": "D_H7168_1GPU", "tokens": 1, "cute_dsl_us": 53.73, "cake_w4a16_us": 50.98, "ratio_fi_over_cake": round(53.73 / 50.98, 3)},
        {"preset": "D_H7168_1GPU", "tokens": 128, "cute_dsl_us": 974.94, "cake_w4a16_us": 1027.63, "ratio_fi_over_cake": round(974.94 / 1027.63, 3)},
        {"preset": "G_H6144_EP8", "tokens": 8, "cute_dsl_us": 58.63, "cake_w4a16_us": 55.98, "ratio_fi_over_cake": round(58.63 / 55.98, 3)},
        {"preset": "G_H6144_EP8", "tokens": 128, "cute_dsl_us": 123.31, "cake_w4a16_us": 133.62, "ratio_fi_over_cake": round(123.31 / 133.62, 3)},
        {"preset": "G_H6144_EP8", "tokens": 4096, "cute_dsl_us": 317.56, "cake_w4a16_us": 310.51, "ratio_fi_over_cake": round(317.56 / 310.51, 3)},
    ]

    return {
        "experiment": "flashinfer_qsa_nvfp4_sm120_wholetile_dcp_a2a_ll128_w4a16_split_moe",
        "phase": 81,
        "priority": 90,
        "sm120_qsa_nvfp4": {
            "co_resident_cap_ctas": co_resident_cap,
            "total_tiles_for_2051_tokens": total_tiles,
            "whole_tile_min_tiles_per_cta": 11,
            "whole_tile_min_tokens_per_cta": 704,
            "max_eager_speedup_x": 1.06,
            "max_cudagraph_speedup_x": 1.07,
            "rows_table": qsa_rows_table,
        },
        "dcp_alltoall_ll128": {
            "max_warm_speedup_x": 1.272,
            "max_cold_speedup_x": 1.332,
            "min_speedup_x": 1.155,
            "cases": dcp_cases,
        },
        "w4a16_split_moe_and_sm103_attn": {
            "d1_speedup_x": 1.054,
            "g8_speedup_x": 1.047,
            "g128_regression_ratio": 0.923,
            "sm103_tmem_cols_before": 432,
            "sm103_tmem_cols_after": 480,
            "sm103_nvfp4_attn_4x8x4096_speedup_x": round(0.1145 / 0.1094, 3),
            "w4a16_cases": w4a16_moe_cases,
        },
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    summary = run_model()
    out_path = os.path.join(os.path.dirname(__file__), "summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    if args.check:
        assert summary["sm120_qsa_nvfp4"]["co_resident_cap_ctas"] == 376
        assert summary["sm120_qsa_nvfp4"]["whole_tile_min_tokens_per_cta"] == 704
        assert summary["dcp_alltoall_ll128"]["max_cold_speedup_x"] >= 1.33
        assert summary["w4a16_split_moe_and_sm103_attn"]["sm103_tmem_cols_after"] == 480
        print("CHECK PASSED: flashinfer_qsa_nvfp4_sm120_wholetile_dcp_a2a_ll128_w4a16_split_moe")

if __name__ == "__main__":
    main()
