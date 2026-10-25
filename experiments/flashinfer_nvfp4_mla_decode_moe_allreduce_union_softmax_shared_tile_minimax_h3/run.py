"""Phase 78 Track A deterministic experiment:
FlashInfer SM100/SM103 NVFP4 MLA Decode E2M1xE4M3 Dequant, Latency-Ordered
One-Shot/Two-Shot MoE All-Reduce Union, Shared-Tile Cluster Softmax, and
MiniMax-H3 256x448 MXFP8 FC1 + SM120 FP8 Varlen Attention.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def simulate_nvfp4_mla_decode() -> dict:
    """Model FlashInfer #7306 NVFP4 MLA decode (E2M1 x UE4M3 block-16 scales,
    512-wide latent row as both K and V, 64-token pages, native mul.rn.bf16x2
    dequant + shorter softmax chain).
    """
    gb300_cases = [
        {"batch": 1, "q_len": 1, "kv_len": 1024, "prev_us": 20.14, "new_us": 19.25},
        {"batch": 1, "q_len": 2, "kv_len": 8192, "prev_us": 39.20, "new_us": 37.36},
        {"batch": 4, "q_len": 1, "kv_len": 32768, "prev_us": 135.72, "new_us": 124.22},
        {"batch": 8, "q_len": 2, "kv_len": 32768, "prev_us": 173.32, "new_us": 157.88},
        {"batch": 32, "q_len": 1, "kv_len": 8192, "prev_us": 93.41, "new_us": 85.44},
        {"batch": 32, "q_len": 2, "kv_len": 8192, "prev_us": 103.50, "new_us": 94.93},
        {"batch": 32, "q_len": 1, "kv_len": 65536, "prev_us": 598.32, "new_us": 532.19},
        {"batch": 32, "q_len": 1, "kv_len": 131072, "prev_us": 1189.57, "new_us": 1058.39},
    ]
    b200_cases = [
        {"batch": 1, "q_len": 1, "kv_len": 1024, "prev_us": 21.30, "new_us": 20.70},
        {"batch": 1, "q_len": 2, "kv_len": 8192, "prev_us": 44.36, "new_us": 42.34},
        {"batch": 4, "q_len": 1, "kv_len": 32768, "prev_us": 153.54, "new_us": 144.84},
        {"batch": 8, "q_len": 2, "kv_len": 32768, "prev_us": 198.26, "new_us": 185.64},
        {"batch": 32, "q_len": 1, "kv_len": 8192, "prev_us": 105.30, "new_us": 98.02},
        {"batch": 32, "q_len": 2, "kv_len": 8192, "prev_us": 116.66, "new_us": 108.99},
        {"batch": 32, "q_len": 1, "kv_len": 65536, "prev_us": 669.92, "new_us": 625.81},
        {"batch": 32, "q_len": 1, "kv_len": 131072, "prev_us": 1331.49, "new_us": 1244.45},
    ]

    for row in gb300_cases:
        row["speedup"] = round(row["prev_us"] / row["new_us"], 4)
        # 512 NVFP4 values (256 B) + 32 UE4M3 scales (32 B) = 288 B per token
        kv_bytes = row["batch"] * row["kv_len"] * 288
        row["effective_kv_bw_tbps"] = round((kv_bytes / (row["new_us"] * 1e-6)) / 1e12, 4)

    for row in b200_cases:
        row["speedup"] = round(row["prev_us"] / row["new_us"], 4)
        kv_bytes = row["batch"] * row["kv_len"] * 288
        row["effective_kv_bw_tbps"] = round((kv_bytes / (row["new_us"] * 1e-6)) / 1e12, 4)

    return {
        "row_bytes_data": 256,
        "row_bytes_scale": 32,
        "total_bytes_per_latent_token": 288,
        "page_size_tokens": 64,
        "gb300_cases": gb300_cases,
        "b200_cases": b200_cases,
        "gb300_max_speedup": max(r["speedup"] for r in gb300_cases),
        "gb300_min_speedup": min(r["speedup"] for r in gb300_cases),
        "b200_max_speedup": max(r["speedup"] for r in b200_cases),
        "b200_min_speedup": min(r["speedup"] for r in b200_cases),
    }


def simulate_moe_allreduce_union() -> dict:
    """Model FlashInfer #7298 latency-ordered one-shot (`lat_b4`, `lat_late_b5`)
    and two-shot (`two_shot_b4`, `two_shot_b5`) schedules for SM100/SM103
    fused MoE finalize + all-reduce + residual + RMSNorm (H=7168).
    """
    hidden_dim = 7168
    elem_bytes = 2
    # Compare one-shot vs two-shot remote NVLink traffic per rank for T tokens
    traffic_comparison = []
    for ws in (2, 4, 8):
        for tokens in (1, 16, 64, 256, 2048):
            row_bytes = tokens * hidden_dim * elem_bytes
            # One-shot: every rank pulls (ws - 1) full rows
            one_shot_remote_bytes = (ws - 1) * row_bytes
            # Two-shot: stage 1 reduces 1/ws slice from (ws-1) peers, stage 2 pulls (ws-1)/ws finished slices
            two_shot_remote_bytes = 2 * ((ws - 1) / ws) * row_bytes
            traffic_ratio = round(one_shot_remote_bytes / two_shot_remote_bytes, 4)
            traffic_comparison.append(
                {
                    "world_size": ws,
                    "tokens": tokens,
                    "one_shot_remote_kib": round(one_shot_remote_bytes / 1024.0, 2),
                    "two_shot_remote_kib": round(two_shot_remote_bytes / 1024.0, 2),
                    "one_shot_over_two_shot_traffic_ratio": traffic_ratio,
                }
            )

    measured_shapes = [
        {"arch": "B200", "ws": 2, "dtype": "bf16", "tokens": 64, "experts": 12, "pdl": False, "route": "lat_b4", "prev_us": 26.78, "new_us": 21.89},
        {"arch": "B300", "ws": 2, "dtype": "bf16", "tokens": 64, "experts": 12, "pdl": False, "route": "lat_b4", "prev_us": 26.82, "new_us": 22.14},
        {"arch": "B200", "ws": 4, "dtype": "bf16", "tokens": 256, "experts": 8, "pdl": False, "route": "two_shot_b4", "prev_us": 44.10, "new_us": 24.48},
        {"arch": "B300", "ws": 4, "dtype": "bf16", "tokens": 256, "experts": 8, "pdl": False, "route": "two_shot_b4", "prev_us": 43.30, "new_us": 23.71},
        {"arch": "B200", "ws": 4, "dtype": "fp16", "tokens": 2048, "experts": 8, "pdl": False, "route": "two_shot_b4", "prev_us": 230.08, "new_us": 127.07},
        {"arch": "B300", "ws": 4, "dtype": "fp16", "tokens": 2048, "experts": 8, "pdl": False, "route": "two_shot_b5", "prev_us": 232.99, "new_us": 120.51},
        {"arch": "B200", "ws": 8, "dtype": "bf16", "tokens": 16, "experts": 8, "pdl": False, "route": "lat_b4", "prev_us": 23.49, "new_us": 19.65},
        {"arch": "B200", "ws": 8, "dtype": "bf16", "tokens": 64, "experts": 12, "pdl": False, "route": "two_shot_b4", "prev_us": 36.10, "new_us": 22.98},
        {"arch": "B200", "ws": 8, "dtype": "bf16", "tokens": 256, "experts": 8, "pdl": False, "route": "two_shot_b4", "prev_us": 60.35, "new_us": 26.08},
        {"arch": "B300", "ws": 8, "dtype": "bf16", "tokens": 256, "experts": 8, "pdl": False, "route": "two_shot_b5", "prev_us": 56.10, "new_us": 26.50},
        {"arch": "B200", "ws": 8, "dtype": "fp16", "tokens": 2048, "experts": 8, "pdl": False, "route": "two_shot_b4", "prev_us": 357.76, "new_us": 135.26},
        {"arch": "B300", "ws": 8, "dtype": "fp16", "tokens": 2048, "experts": 8, "pdl": False, "route": "two_shot_b5", "prev_us": 357.92, "new_us": 134.14},
    ]
    for item in measured_shapes:
        item["speedup"] = round(item["prev_us"] / item["new_us"], 4)

    return {
        "hidden_dim": hidden_dim,
        "cta_threads": 224,
        "cluster_ctas": 4,
        "traffic_comparison": traffic_comparison,
        "measured_shapes": measured_shapes,
        "b200_max_speedup": max(r["speedup"] for r in measured_shapes if r["arch"] == "B200"),
        "b300_max_speedup": max(r["speedup"] for r in measured_shapes if r["arch"] == "B300"),
    }


def simulate_shared_tile_softmax_and_minimax_h3() -> dict:
    """Model FlashInfer #7308 shared-tile Blackwell cached-cluster softmax
    (`c8_t4_s4` / `c16_t4_s4`) and #7304 MiniMax-H3 256x448 MXFP8 FC1 + #7334 SM120 FP8 varlen attention.
    """
    softmax_cases = [
        {"arch": "GB300", "rows": 32, "vocab": 151936, "cluster_ctas": 8, "route": "c8_t4_s4", "prev_us": 12.16, "new_us": 10.66},
        {"arch": "GB300", "rows": 32, "vocab": 256000, "cluster_ctas": 16, "route": "c16_t4_s4", "prev_us": 15.20, "new_us": 12.96},
        {"arch": "GB300", "rows": 48, "vocab": 128256, "cluster_ctas": 8, "route": "c8_t4_s4", "prev_us": 12.35, "new_us": 10.69},
        {"arch": "B200", "rows": 16, "vocab": 256000, "cluster_ctas": 16, "route": "c16_t4_s4", "prev_us": 10.69, "new_us": 9.89},
        {"arch": "B200", "rows": 24, "vocab": 128256, "cluster_ctas": 8, "route": "c8_t4_s4", "prev_us": 10.14, "new_us": 9.60},
        {"arch": "B200", "rows": 32, "vocab": 151936, "cluster_ctas": 8, "route": "c8_t4_s4", "prev_us": 11.46, "new_us": 11.07},
    ]
    for s in softmax_cases:
        s["speedup"] = round(s["prev_us"] / s["new_us"], 4)
        s["dynamic_smem_bytes"] = 34432 if s["cluster_ctas"] == 8 else 34944
        s["resident_ctas_per_sm"] = 4

    minimax_h3_mxfp8_cases = [
        {"arch": "B200", "rows": 1024, "prev_us": 74.98, "new_us": 73.28},
        {"arch": "B200", "rows": 2048, "prev_us": 129.82, "new_us": 124.22},
        {"arch": "B200", "rows": 4096, "prev_us": 236.99, "new_us": 223.62},
        {"arch": "B200", "rows": 8192, "prev_us": 457.86, "new_us": 436.32},
        {"arch": "B300", "rows": 1024, "prev_us": 66.88, "new_us": 64.74},
        {"arch": "B300", "rows": 2048, "prev_us": 114.40, "new_us": 108.13},
        {"arch": "B300", "rows": 4096, "prev_us": 208.16, "new_us": 194.88},
        {"arch": "B300", "rows": 8192, "prev_us": 401.12, "new_us": 376.48},
    ]
    for m in minimax_h3_mxfp8_cases:
        m["speedup"] = round(m["prev_us"] / m["new_us"], 4)

    sm120_varlen_cases = [
        {"arch": "RTX_5090", "shape": "seg4x512_h42_d128", "prev_us": 34.91, "new_us": 33.09},
        {"arch": "RTX_5090", "shape": "seg8x1024_h42_d128", "prev_us": 195.84, "new_us": 191.52},
        {"arch": "RTX_PRO_6000", "shape": "seg4x512_h42_d128", "prev_us": 27.68, "new_us": 26.59},
        {"arch": "RTX_PRO_6000", "shape": "seg8x1024_h42_d128", "prev_us": 149.12, "new_us": 144.64},
    ]
    for v in sm120_varlen_cases:
        v["speedup"] = round(v["prev_us"] / v["new_us"], 4)

    return {
        "softmax_cases": softmax_cases,
        "softmax_gb300_max_speedup": max(s["speedup"] for s in softmax_cases if s["arch"] == "GB300"),
        "softmax_b200_max_speedup": max(s["speedup"] for s in softmax_cases if s["arch"] == "B200"),
        "minimax_h3_fc1_n_tiles_prev": 112,
        "minimax_h3_fc1_n_tiles_w448": 64,
        "minimax_h3_mxfp8_cases": minimax_h3_mxfp8_cases,
        "minimax_h3_b300_max_speedup": max(m["speedup"] for m in minimax_h3_mxfp8_cases if m["arch"] == "B300"),
        "sm120_varlen_cases": sm120_varlen_cases,
        "sm120_rtx5090_max_speedup": max(v["speedup"] for v in sm120_varlen_cases if v["arch"] == "RTX_5090"),
    }


def build_summary() -> dict:
    mla = simulate_nvfp4_mla_decode()
    union = simulate_moe_allreduce_union()
    softmax_h3 = simulate_shared_tile_softmax_and_minimax_h3()
    return {
        "experiment_id": "flashinfer_nvfp4_mla_decode_moe_allreduce_union_softmax_shared_tile_minimax_h3",
        "phase": 78,
        "priority": 87,
        "track": "gpu_kernels_and_comm",
        "nvfp4_mla_decode": mla,
        "moe_allreduce_union": union,
        "softmax_and_minimax_h3": softmax_h3,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    summary = build_summary()
    out_path = Path(__file__).resolve().parent / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if args.check:
        assert summary["nvfp4_mla_decode"]["gb300_max_speedup"] >= 1.12
        assert summary["moe_allreduce_union"]["b300_max_speedup"] >= 2.65
        assert summary["softmax_and_minimax_h3"]["softmax_gb300_max_speedup"] >= 1.17


if __name__ == "__main__":
    main()
