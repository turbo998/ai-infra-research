from __future__ import annotations

import json
import math
from pathlib import Path


def run_simulation() -> dict:
    # 1. cake_grouped_fp8_gemm swap-AB block-scaled UE8M0 programs (84f3bb30b29e)
    grouped_fp8_cases = [
        {
            "shape": "16000x512x4096_g512_a32_t2x10",
            "program": "swap_run32",
            "b200_shipped_us": 42.30,
            "b200_new_us": 24.58,
            "gb300_fast_shipped_us": 39.90,
            "gb300_fast_new_us": 23.52,
            "gb300_slow_shipped_us": 40.56,
            "gb300_slow_new_us": 23.49,
        },
        {
            "shape": "8064x1024x2048_g256_a32_t8x8",
            "program": "swap_run32",
            "b200_shipped_us": 28.58,
            "b200_new_us": 23.14,
            "gb300_fast_shipped_us": 27.95,
            "gb300_fast_new_us": 22.53,
            "gb300_slow_shipped_us": 27.65,
            "gb300_slow_new_us": 22.59,
        },
        {
            "shape": "67584x4096x256_g512_a128_t256x10",
            "program": "swap_carry",
            "b200_shipped_us": 113.38,
            "b200_new_us": 101.78,
            "gb300_fast_shipped_us": 110.26,
            "gb300_fast_new_us": 97.87,
            "gb300_slow_shipped_us": 109.36,
            "gb300_slow_new_us": 97.95,
        },
        {
            "shape": "105984x512x4096_g512_a128_t4096x10",
            "program": "swap",
            "b200_shipped_us": 224.34,
            "b200_new_us": 203.36,
            "gb300_fast_shipped_us": 220.38,
            "gb300_fast_new_us": 197.74,
            "gb300_slow_shipped_us": 218.22,
            "gb300_slow_new_us": 197.81,
        },
    ]
    for c in grouped_fp8_cases:
        c["b200_ratio"] = round(c["b200_new_us"] / c["b200_shipped_us"], 3)
        c["b200_speedup"] = round(c["b200_shipped_us"] / c["b200_new_us"], 3)
        c["gb300_fast_ratio"] = round(c["gb300_fast_new_us"] / c["gb300_fast_shipped_us"], 3)
        c["gb300_slow_ratio"] = round(c["gb300_slow_new_us"] / c["gb300_slow_shipped_us"], 3)

    # 2. nvfp4_sparse_mla_decode tcgen05 programs (a8ec053888bd)
    cluster_sizes = [1, 2, 3, 4, 5, 6, 8]
    split_bounds = [64, 128, 192, 256, 999999]
    total_tcgen05_programs = len(cluster_sizes) * len(split_bounds)
    mla_rows = [
        {"row": "P_T5_topk2048", "tokens": 5, "topk": 2048, "b200_prev_us": 12.32, "b200_new_us": 10.02, "gb300_prev_us": 11.62, "gb300_new_us": 9.57},
        {"row": "P_T30_topk2048", "tokens": 30, "topk": 2048, "b200_prev_us": 20.29, "b200_new_us": 16.26, "gb300_prev_us": 18.72, "gb300_new_us": 15.14},
        {"row": "W_T46_topk2048", "tokens": 46, "topk": 2048, "b200_prev_us": 31.97, "b200_new_us": 24.77, "gb300_prev_us": 23.71, "gb300_new_us": 19.04},
        {"row": "W_T128_topk2048", "tokens": 128, "topk": 2048, "b200_prev_us": 63.94, "b200_new_us": 48.99, "gb300_prev_us": 61.60, "gb300_new_us": 44.80},
    ]
    for r in mla_rows:
        r["b200_speedup"] = round(r["b200_prev_us"] / r["b200_new_us"], 3)
        r["gb300_speedup"] = round(r["gb300_prev_us"] / r["gb300_new_us"], 3)

    # 3. CakeMxfp8MegaMoeEp16 pushed routes and rows (38b6198055ca)
    world_size = 16
    tokens_per_rank = 64
    hidden_dim = 3072
    top_k = 8
    rows_mailbox_bytes = world_size * tokens_per_rank * hidden_dim * 2  # BF16
    routes_mailbox_bytes = world_size * tokens_per_rank * top_k * 8  # id + weight
    ep16_cases = [
        {"tokens_per_rank": 16, "routing": "balanced", "prev_us": 299.9, "new_us": 295.4, "speedup": 1.015},
        {"tokens_per_rank": 32, "routing": "hotset50", "prev_us": 301.6, "new_us": 296.2, "speedup": 1.018},
        {"tokens_per_rank": 64, "routing": "hotset50", "prev_us": 305.0, "new_us": 299.3, "speedup": 1.019},
    ]

    # 4. MiniMax-H3 SM100/SM103/SM120 out-proj & varlen attention (b4125442a533, 53ecbde224f4, b8ed186df31b)
    minimax_h3 = {
        "sm100_nvfp4_tail_split_p8_m4824": {
            "gb300_before_us": 82.8,
            "gb300_after_us": 79.5,
            "gb300_speedup": 1.041,
            "gb200_before_us": 98.2,
            "gb200_after_us": 93.6,
            "gb200_speedup": 1.049,
            "gb300_geomean_20_shapes": 1.021,
            "gb200_geomean_7_shapes": 1.042,
        },
        "sm120_fp8_out_proj_m48768": {
            "before_us": 5285.4,
            "after_us": 5225.9,
            "speedup": 1.011,
            "geomean_6_shapes": 1.009,
        },
        "sm120_fp8_varlen_attn": {
            "rtx5090_seg_empty_prev_us": 46.8,
            "rtx5090_seg_empty_new_us": 44.3,
            "rtx5090_seg_empty_speedup": 1.055,
            "rtx_pro_6000_center_4s_prev_us": 48276.0,
            "rtx_pro_6000_center_4s_new_us": 46558.0,
            "rtx_pro_6000_center_4s_speedup": 1.037,
        },
    }

    return {
        "experiment": "flashinfer_cake_fp8_gemm_swap_ab_nvfp4_mla_tcgen05_mxfp8_ep16_push_minimax_h3",
        "evidence_class": "own-offline",
        "grouped_fp8_gemm_swap_ab": {
            "programs": ["swap", "swap_run32", "swap_carry"],
            "min_b200_ratio": 0.581,
            "min_gb300_slow_ratio": 0.579,
            "max_speedup_b200": round(42.30 / 24.58, 3),
            "cases": grouped_fp8_cases,
        },
        "nvfp4_sparse_mla_decode_tcgen05": {
            "cluster_sizes": cluster_sizes,
            "split_bounds": [64, 128, 192, 256, "any"],
            "total_tcgen05_programs": total_tcgen05_programs,
            "kv_row_bytes": 352,
            "b200_geomean_speedup": 1.131,
            "gb300_geomean_speedup": 1.152,
            "gb300_max_speedup": 1.375,
            "rows": mla_rows,
        },
        "mxfp8_megamoe_ep16_push": {
            "world_size": world_size,
            "hidden_dim": hidden_dim,
            "intermediate_dim": 5120,
            "num_experts": 512,
            "top_k": top_k,
            "rows_mailbox_bytes_t64": rows_mailbox_bytes,
            "routes_mailbox_bytes_t64": routes_mailbox_bytes,
            "max_session_forwards_before_recreate": 14913080,
            "cases": ep16_cases,
        },
        "minimax_h3_family": minimax_h3,
    }


def main() -> None:
    out = run_simulation()
    out_path = Path(__file__).with_name("summary.json")
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
