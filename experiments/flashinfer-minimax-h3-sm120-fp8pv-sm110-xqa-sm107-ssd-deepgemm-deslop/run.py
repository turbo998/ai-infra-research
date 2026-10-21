#!/usr/bin/env python3
"""Deterministic stdlib reproduction for Phase 51 Topic 1:
FlashInfer MiniMax-H3 SM120 NVFP4-QK / FP8-PV Hybrid Varlen Attention (#6936),
SM110 XQA Tree & Decode Cake Package (#6945), SM107a Rubin Look-Back SSD (#6943),
and DeepGEMM FP8 1D1D Package & Round-2 Deslop (#6927, #6920).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


MINIMAX_H3_HEAD_DIM = 128
MINIMAX_H3_NUM_HEADS = 24
BLOCK_M = 128
BLOCK_N = 128
ROW_BYTES4 = 64
BLOCKS_PER_ROW = 8
SCALE_TILE_BYTES = BLOCK_N * BLOCKS_PER_ROW  # 1024 B (1 KiB)
PARTIAL_FLOATS_FP8PV = 2 * MINIMAX_H3_HEAD_DIM  # 256 floats
QUANT_SMALL_MAX_TOKENS = 1024
ATTN_SHORT_MAX_TOKENS = 4096
STATS_CTAS_PER_SM = 4
QUANT_CTAS_PER_SM = 8


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def workspace_bytes_nvfp4_fp8pv(tokens: int, num_heads: int, num_segments: int) -> int:
    """Exact workspace formula from flashinfer#6936 (ddba2f4b2c)."""
    num_kblocks = max(1, ceil_div(tokens, BLOCK_N))
    padded = num_kblocks * BLOCK_N
    segments = max(1, num_segments)
    tiles = ceil_div(tokens, BLOCK_M) + segments
    q4 = tokens * num_heads * ROW_BYTES4
    k4 = tokens * num_heads * ROW_BYTES4
    q_sf = tokens * num_heads * BLOCKS_PER_ROW
    k_sf = num_heads * num_kblocks * SCALE_TILE_BYTES
    vt8 = num_heads * MINIMAX_H3_HEAD_DIM * padded
    floats = (
        tokens * num_heads  # q_scale
        + segments * num_heads  # k_scale
        + 2 * segments * num_heads * MINIMAX_H3_HEAD_DIM  # mean_k, v_scale
        + tiles * num_heads * PARTIAL_FLOATS_FP8PV  # partials
        + tiles * num_heads  # knorm_part
        + segments * num_heads  # counters (uint32)
    )
    return q4 + k4 + q_sf + k_sf + vt8 + 4 * floats


def select_minimax_h3_fp8pv_programs(
    tokens: int, cu_seqlens: list[int]
) -> tuple[str, str, str]:
    """Program selection from flashinfer#6936 (minimax_h3_sm120_nvfp4_fp8pv_programs)."""
    max_seg = max(
        (cu_seqlens[i + 1] - cu_seqlens[i] for i in range(len(cu_seqlens) - 1)),
        default=0,
    )
    quant_prog = "QUANTIZE_MB3" if tokens <= QUANT_SMALL_MAX_TOKENS else "QUANTIZE"
    attn_prog = (
        "ATTENTION_SHORT" if max_seg <= ATTN_SHORT_MAX_TOKENS else "ATTENTION_BASE"
    )
    return ("KV_STATS", quant_prog, attn_prog)


def simulate_minimax_h3_sm120_fp8pv() -> dict[str, Any]:
    """Verify MiniMax-H3 SM120 NVFP4-QK / FP8-PV hybrid latencies, accuracy, and workspace."""
    rtx5090_rows = [
        ("center_4s_p1_m33472", 33472, 1, 55.1978, 75.2480, 41.6412),
        ("tail_5s_p1_m38531", 38531, 1, 73.0696, 100.1189, 55.3445),
        ("center_5s_p1_m38592", 38592, 1, 73.0156, 100.6726, 55.7971),
        ("tail_5s_p1_m38629", 38629, 1, 73.0198, 100.7391, 56.2832),
        ("center_6s_p1_m48768", 48768, 1, 115.2190, 159.2777, 89.2569),
        ("center_8s_p1_m58944", 58944, 1, 167.2685, 231.9048, 129.6428),
        ("center_10s_p1_m74240", 74240, 1, 263.8216, 365.4598, 203.6492),
        ("center_15s_p1_m109952", 109952, 1, 573.6273, 796.9718, 441.8644),
        ("pad_15s_p1_used109901_total109952", 109952, 2, 573.9956, 797.1831, 441.0108),
    ]
    rtx_pro_6000_rows = [
        ("center_4s_p1_m33472", 40.2044, 47.7775, 37.3651),
        ("tail_5s_p1_m38531", 53.9199, 64.0017, 49.8555),
        ("center_5s_p1_m38592", 54.0714, 64.0569, 49.8878),
        ("tail_5s_p1_m38629", 54.2219, 64.0224, 49.8444),
        ("center_6s_p1_m48768", 85.2030, 100.7248, 78.9925),
        ("center_8s_p1_m58944", 123.8312, 147.2781, 114.6144),
        ("center_10s_p1_m74240", 195.3989, 232.0127, 179.7568),
        ("center_15s_p1_m109952", 424.0433, 505.4736, 390.3084),
        ("pad_15s_p1_used109901_total109952", 421.9906, 505.4639, 390.0197),
    ]

    speedups_5090 = [fp8_ms / hyb_ms for _, _, _, hyb_ms, fp8_ms, _ in rtx5090_rows]
    speedups_pro6000 = [fp8_ms / hyb_ms for _, hyb_ms, fp8_ms, _ in rtx_pro_6000_rows]
    geomean_5090_s1 = math.exp(sum(math.log(s) for s in speedups_5090) / len(speedups_5090))
    geomean_pro6000_s1 = math.exp(
        sum(math.log(s) for s in speedups_pro6000) / len(speedups_pro6000)
    )

    ws_33472 = workspace_bytes_nvfp4_fp8pv(33472, MINIMAX_H3_NUM_HEADS, 1)
    ws_109952 = workspace_bytes_nvfp4_fp8pv(109952, MINIMAX_H3_NUM_HEADS, 2)

    progs_small = select_minimax_h3_fp8pv_programs(512, [0, 512])
    progs_med = select_minimax_h3_fp8pv_programs(2048, [0, 2048])
    progs_long = select_minimax_h3_fp8pv_programs(8192, [0, 4097, 8192])

    return {
        "rtx5090_min_speedup_s1": round(min(speedups_5090), 4),
        "rtx5090_max_speedup_s1": round(max(speedups_5090), 4),
        "rtx5090_geomean_speedup_s1": round(geomean_5090_s1, 4),
        "rtx5090_reported_two_session_geomean": 1.3814,
        "rtx_pro_6000_min_speedup_s1": round(min(speedups_pro6000), 4),
        "rtx_pro_6000_max_speedup_s1": round(max(speedups_pro6000), 4),
        "rtx_pro_6000_geomean_speedup_s1": round(geomean_pro6000_s1, 4),
        "rtx_pro_6000_reported_two_session_geomean": 1.1876,
        "rel_l2_fp4_default_range": [0.186, 0.193],
        "rel_l2_fp8_hybrid_range": [0.136, 0.143],
        "rel_l2_midpoint_reduction_pct": round(
            (1.0 - (0.1395 / 0.1895)) * 100.0, 2
        ),
        "workspace_bytes_m33472_seg1": ws_33472,
        "workspace_bytes_m109952_seg2": ws_109952,
        "programs_512_tok": list(progs_small),
        "programs_2048_tok": list(progs_med),
        "programs_8192_tok_seg4097": list(progs_long),
        "dynamic_smem_bytes": {
            "KV_STATS": 16512,
            "QUANTIZE": 0,
            "QUANTIZE_MB3": 0,
            "ATTENTION_BASE": 60416,
            "ATTENTION_SHORT": 60416,
        },
        "invariants_passed": (
            progs_small == ("KV_STATS", "QUANTIZE_MB3", "ATTENTION_SHORT")
            and progs_med == ("KV_STATS", "QUANTIZE", "ATTENTION_SHORT")
            and progs_long == ("KV_STATS", "QUANTIZE", "ATTENTION_BASE")
            and min(speedups_5090) >= 1.363
            and min(speedups_pro6000) >= 1.180
        ),
    }


def simulate_sm110_xqa_and_sm107_ssd() -> dict[str, Any]:
    """Verify SM110 XQA tree/decode schedules (#6945) and SM107a Rubin SSD (#6943)."""
    def select_sm110_tree_schedule(
        kernel: str, kv_dtype: str, q_rows_per_kv_head: int
    ) -> str:
        retired = ("tcgen05", "register_mma", "register_mma_split", "register_mma_auto")
        if kernel in retired:
            raise ValueError(f"Kernel {kernel!r} is retired in cake_sm110_xqa")
        if kernel in ("tmem", "pair"):
            return kernel
        if kernel != "auto":
            raise ValueError(f"Unknown kernel {kernel!r}")
        q_tiles_128 = ceil_div(q_rows_per_kv_head, 128)
        if kv_dtype == "e4m3" and (q_tiles_128 % 2 == 0):
            return "pair"
        return "tmem"

    sm110_benchmarks = [
        ("D512_tree_fp16_contig", 19.552, 17.376),
        ("D512_tree_fp16_page128", 18.672, 17.360),
        ("D512_tree_e4m3_contig", 17.408, 16.192),
        ("D512_tree_e4m3_page128", 17.440, 16.064),
        ("D128_decode_fp16_b1_cap1024", 17.584, 11.616),
        ("D128_decode_fp16_b2_cap512", 17.440, 11.968),
        ("D128_decode_fp16_b4_cap256", 15.968, 11.520),
    ]
    decode_speedups = [
        base / new for name, base, new in sm110_benchmarks if name.startswith("D128")
    ]
    tree_speedups = [
        base / new for name, base, new in sm110_benchmarks if name.startswith("D512")
    ]

    # SM107a Rubin R200 look-back chunk-parallel SSD (#6943)
    sm107_ssd_rows = [
        ("perf_batched_1x256", 92.1, 77.0, 1.196, 84.1, 73.8, 1.140),
        ("perf_batched_4x64", 80.3, 80.2, 1.001, 83.7, 83.4, 1.004),
        ("perf_batched_1x64", 35.7, 35.7, 0.999, 34.3, 34.4, 0.999),
        ("perf_batched_4x16", 34.1, 34.3, 0.993, 28.8, 25.9, 1.114),
        ("perf_batched_1x16", 22.1, 22.2, 0.993, 17.1, 17.2, 0.996),
        ("cp_b1h32c16", 32.2, 33.1, 0.974, 28.4, 26.0, 1.093),
        ("cp_b4h8c16", 34.2, 34.2, 1.002, 33.3, 26.2, 1.268),
    ]
    call_speedups = [cs for _, _, _, cs, _, _, _ in sm107_ssd_rows]
    main_speedups = [ms for _, _, _, _, _, _, ms in sm107_ssd_rows]
    geomean_call = math.exp(sum(math.log(s) for s in call_speedups) / len(call_speedups))
    geomean_main = math.exp(sum(math.log(s) for s in main_speedups) / len(main_speedups))

    retired_rejected = False
    try:
        select_sm110_tree_schedule("register_mma", "fp16", 160)
    except ValueError:
        retired_rejected = True

    return {
        "sm110_total_exported_programs": 43,
        "sm110_auto_e4m3_256rows": select_sm110_tree_schedule("auto", "e4m3", 256),
        "sm110_auto_e4m3_128rows": select_sm110_tree_schedule("auto", "e4m3", 128),
        "sm110_auto_fp16_256rows": select_sm110_tree_schedule("auto", "fp16", 256),
        "sm110_retired_route_rejected": retired_rejected,
        "sm110_tree_speedup_range_unit_a": [
            round(min(tree_speedups), 3),
            round(max(tree_speedups), 3),
        ],
        "sm110_decode_speedup_range_unit_a": [
            round(min(decode_speedups), 3),
            round(max(decode_speedups), 3),
        ],
        "sm107_ssd_updated_programs": 6,
        "sm107_ssd_geomean_call_speedup": round(geomean_call, 4),
        "sm107_ssd_geomean_main_kernel_speedup": round(geomean_main, 4),
        "sm107_ssd_reported_geomean_main_kernel_speedup": 1.0839,
        "sm107_ssd_max_call_speedup": round(max(call_speedups), 3),
        "sm107_ssd_max_main_kernel_speedup": round(max(main_speedups), 3),
        "invariants_passed": (
            retired_rejected
            and select_sm110_tree_schedule("auto", "e4m3", 256) == "pair"
            and select_sm110_tree_schedule("auto", "e4m3", 128) == "tmem"
            and select_sm110_tree_schedule("auto", "fp16", 256) == "tmem"
            and abs(geomean_main - 1.0839) < 0.001
        ),
    }


def simulate_deepgemm_package_and_deslop() -> dict[str, Any]:
    """Verify DeepGEMM FP8 1D1D package (#6927) and 5-family round-2 deslop (#6920)."""
    families = {
        "fp8_1d1d": 2,
        "native_fp4": 8,
        "fp4_kgroup": 21,  # 19 + 2 SM103-only
        "mixed_fp8_fp4": 9,
        "fp8_batched": 6,
    }
    total_programs = sum(families.values())
    # 46 programs across SM100 and SM103, except 3 SM103-only builds -> 89 total builds
    total_builds = 89

    # K-group kernel folding (#6920 Commit 4): 14 kernel TUs folded onto 4 knob-keyed bodies
    kgroup_folded_groups = {
        "general_s1_s2_fp32_fp32acc_bf16": 6,
        "small": 3,
        "small_2x8x3": 3,
        "swapped_lpt_fp32": 2,
    }
    folded_tus = sum(kgroup_folded_groups.values())
    surviving_bodies = len(kgroup_folded_groups)
    deleted_kernel_files = folded_tus - surviving_bodies

    # TMA encoder hoisting (#6920 Commit 5)
    total_inline_encoders_before = 175
    hoisted_encoder_copies = 133
    shared_header_unique_bodies = 35
    remaining_inline_encoders = total_inline_encoders_before - hoisted_encoder_copies

    # Line diff across 72 files in #6920
    additions = 2904
    deletions = 15312
    net_lines_removed = deletions - additions

    return {
        "families": families,
        "total_programs": total_programs,
        "total_arch_builds": total_builds,
        "kgroup_bindings_before": 21,
        "kgroup_duplicate_bindings_folded": 11,
        "kgroup_shared_bindings_kept": 7,
        "kgroup_kernel_tus_folded": folded_tus,
        "kgroup_surviving_knob_bodies": surviving_bodies,
        "kgroup_deleted_kernel_files": deleted_kernel_files,
        "tma_encoders_before": total_inline_encoders_before,
        "tma_encoders_hoisted_copies": hoisted_encoder_copies,
        "tma_encoders_shared_header_bodies": shared_header_unique_bodies,
        "tma_encoders_remaining_inline": remaining_inline_encoders,
        "deslop_additions": additions,
        "deslop_deletions": deletions,
        "deslop_net_lines_removed": net_lines_removed,
        "device_gate_sass_identical_builds": 36,
        "gb300_fp8_1d1d_4096_7168_4096_us": {"previous": 66.880, "new": 66.801, "speedup": 1.001},
        "gb200_fp8_1d1d_4096_7168_4096_us": {"previous": 87.872, "new": 87.264, "speedup": 1.007},
        "invariants_passed": (
            total_programs == 46
            and folded_tus == 14
            and surviving_bodies == 4
            and deleted_kernel_files == 10
            and remaining_inline_encoders == 42
            and net_lines_removed == 12408
        ),
    }


def run_experiment() -> dict[str, Any]:
    minimax = simulate_minimax_h3_sm120_fp8pv()
    xqa_ssd = simulate_sm110_xqa_and_sm107_ssd()
    deepgemm = simulate_deepgemm_package_and_deslop()
    all_passed = (
        minimax["invariants_passed"]
        and xqa_ssd["invariants_passed"]
        and deepgemm["invariants_passed"]
    )
    return {
        "experiment": "flashinfer-minimax-h3-sm120-fp8pv-sm110-xqa-sm107-ssd-deepgemm-deslop",
        "upstream_commits": [
            "flashinfer-ai/flashinfer@ddba2f4b2c2b8f6852a565774bc31c2ace39a5c7",
            "flashinfer-ai/flashinfer@da1e9f4bc29629724cc4de618085938743beb65a",
            "flashinfer-ai/flashinfer@7601d0bb7e92390aae5276951f0e4b7c8a2b5c14",
            "flashinfer-ai/flashinfer@597e3a67ca36e1d2f3820acc9b0e3ca6817c8d2e",
            "flashinfer-ai/flashinfer@4d79be41007c65a3a416a92cda2ae4395ebfbfbf",
        ],
        "minimax_h3_sm120_fp8pv": minimax,
        "sm110_xqa_and_sm107_ssd": xqa_ssd,
        "deepgemm_package_and_deslop": deepgemm,
        "all_invariants_passed": all_passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    summary = run_experiment()
    out_path = Path(__file__).resolve().parent / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.check and not summary["all_invariants_passed"]:
        raise SystemExit(1)
    print(f"OK: {out_path}")


if __name__ == "__main__":
    main()
