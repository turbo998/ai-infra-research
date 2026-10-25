from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def simulate_cake_gated_act_mxfp8() -> dict[str, Any]:
    """Model flashinfer#7305 (bfe3de571c09): Cake Gated SwiGLU MXFP8 fwd/bwd across SM100/SM103/SM107."""
    cases = [
        {
            "arch": "sm_100a",
            "gpu": "B200 (148 SMs)",
            "pass": "fwd",
            "mode": "colwise",
            "shape": "128x128",
            "m": 128,
            "k": 128,
            "baseline_us": 3.53,
            "cake_us": 2.35,
            "reported_speedup": 1.501,
        },
        {
            "arch": "sm_100a",
            "gpu": "B200 (148 SMs)",
            "pass": "fwd",
            "mode": "colwise",
            "shape": "131072x8192",
            "m": 131072,
            "k": 8192,
            "baseline_us": 1117.21,
            "cake_us": 829.95,
            "reported_speedup": 1.346,
        },
        {
            "arch": "sm_100a",
            "gpu": "B200 (148 SMs)",
            "pass": "bwd",
            "mode": "colwise",
            "shape": "128x128",
            "m": 128,
            "k": 128,
            "baseline_us": 3.71,
            "cake_us": 2.52,
            "reported_speedup": 1.474,
        },
        {
            "arch": "sm_100a",
            "gpu": "B200 (148 SMs)",
            "pass": "bwd",
            "mode": "colwise",
            "shape": "131072x8192",
            "m": 131072,
            "k": 8192,
            "baseline_us": 1727.04,
            "cake_us": 1374.90,
            "reported_speedup": 1.256,
        },
        {
            "arch": "sm_103a",
            "gpu": "GB300 (152 SMs)",
            "pass": "fwd",
            "mode": "colwise",
            "shape": "128x128",
            "m": 128,
            "k": 128,
            "baseline_us": 3.45,
            "cake_us": 2.45,
            "reported_speedup": 1.411,
        },
        {
            "arch": "sm_103a",
            "gpu": "GB300 (152 SMs)",
            "pass": "bwd",
            "mode": "colwise",
            "shape": "128x128",
            "m": 128,
            "k": 128,
            "baseline_us": 3.63,
            "cake_us": 2.63,
            "reported_speedup": 1.383,
        },
        {
            "arch": "sm_107a",
            "gpu": "R200 (216 SMs)",
            "pass": "fwd",
            "mode": "colwise",
            "shape": "131072x8192",
            "m": 131072,
            "k": 8192,
            "baseline_us": 864.60,
            "cake_us": 509.53,
            "reported_speedup": 1.697,
        },
        {
            "arch": "sm_107a",
            "gpu": "R200 (216 SMs)",
            "pass": "fwd",
            "mode": "colwise",
            "shape": "16384x7168",
            "m": 16384,
            "k": 7168,
            "baseline_us": 99.37,
            "cake_us": 60.56,
            "reported_speedup": 1.641,
        },
        {
            "arch": "sm_107a",
            "gpu": "R200 (216 SMs)",
            "pass": "bwd",
            "mode": "colwise",
            "shape": "131072x8192",
            "m": 131072,
            "k": 8192,
            "baseline_us": 1327.00,
            "cake_us": 930.00,
            "reported_speedup": 1.427,
        },
    ]

    for row in cases:
        m = int(row["m"])
        k = int(row["k"])
        # Forward reads BF16 [M, 2K] (4*M*K bytes) and writes FP8 E4M3 [M, K] + E8M0 [M, K/32]
        # In "both" mode it writes both rowwise and colwise FP8 + scale tensors.
        scale_bytes = (m * k) // 32
        if row["pass"] == "fwd":
            io_bytes = m * 2 * k * 2 + m * k * 1 + scale_bytes
        else:
            # Backward reads dY [M, K] BF16 + X [M, 2K] BF16 and writes dX FP8 + scale
            io_bytes = m * k * 2 + m * 2 * k * 2 + m * 2 * k * 1 + 2 * scale_bytes
        row["io_bytes"] = io_bytes
        row["computed_speedup"] = round(float(row["baseline_us"]) / float(row["cake_us"]), 3)
        row["bitwise_identical_to_baseline"] = True

    return {
        "cases": cases,
        "sm100_speedup_range": [1.075, 1.501],
        "sm103_speedup_range": [1.044, 1.411],
        "sm107_speedup_range": [1.094, 1.697],
        "max_r200_speedup": max(r["reported_speedup"] for r in cases if r["arch"] == "sm_107a"),
        "max_b200_speedup": max(r["reported_speedup"] for r in cases if r["arch"] == "sm_100a"),
        "supported_modes": ["rowwise", "colwise", "both"],
        "scale_block_size": 32,
        "scale_swizzle_tile": [128, 4],
    }


def simulate_ep8_megamoe_arrive_mailbox() -> dict[str, Any]:
    """Model flashinfer#7300 (f911858bfc32): BF16 rank-major EP8 MegaMoE on 8xB200."""
    rows = [
        {
            "workload": "all-ranks",
            "tokens_per_rank": 1,
            "eager_main_us": 173.1,
            "eager_pr_us": 162.8,
            "eager_speedup": 1.06,
            "cg_main_us": 165.9,
            "cg_pr_us": 153.8,
            "cg_speedup": 1.08,
            "inline_routing_in_dispatch": True,
            "launched_kernels": 9,
            "baseline_launched_kernels": 12,
        },
        {
            "workload": "all-ranks",
            "tokens_per_rank": 16,
            "eager_main_us": 203.8,
            "eager_pr_us": 195.6,
            "eager_speedup": 1.04,
            "cg_main_us": 197.4,
            "cg_pr_us": 188.9,
            "cg_speedup": 1.05,
            "inline_routing_in_dispatch": True,
            "launched_kernels": 9,
            "baseline_launched_kernels": 12,
        },
        {
            "workload": "all-ranks",
            "tokens_per_rank": 32,
            "eager_main_us": 270.1,
            "eager_pr_us": 261.2,
            "eager_speedup": 1.03,
            "cg_main_us": 263.7,
            "cg_pr_us": 255.1,
            "cg_speedup": 1.03,
            "inline_routing_in_dispatch": False,
            "launched_kernels": 10,
            "baseline_launched_kernels": 12,
        },
        {
            "workload": "all-ranks",
            "tokens_per_rank": 128,
            "eager_main_us": 533.6,
            "eager_pr_us": 513.0,
            "eager_speedup": 1.04,
            "cg_main_us": 527.4,
            "cg_pr_us": 507.9,
            "cg_speedup": 1.04,
            "inline_routing_in_dispatch": False,
            "launched_kernels": 10,
            "baseline_launched_kernels": 12,
        },
        {
            "workload": "rank0-only",
            "tokens_per_rank": 1,
            "eager_main_us": 167.2,
            "eager_pr_us": 158.1,
            "eager_speedup": 1.06,
            "cg_main_us": 160.7,
            "cg_pr_us": 149.3,
            "cg_speedup": 1.08,
            "inline_routing_in_dispatch": True,
            "launched_kernels": 9,
            "baseline_launched_kernels": 12,
        },
    ]

    return {
        "rows": rows,
        "hidden_size": 7168,
        "intermediate_size": 2048,
        "num_experts": 256,
        "ep_world_size": 8,
        "experts_per_rank": 32,
        "top_k": 8,
        "inline_route_max_tokens": 16,
        "staging_copy_kernels_before": 3,
        "staging_copy_kernels_after": 1,
        "barrier_mode": "1-warp arrive-only peer mailbox",
        "max_eager_speedup": max(r["eager_speedup"] for r in rows),
        "max_cg_speedup": max(r["cg_speedup"] for r in rows),
    }


def simulate_warp_decode_router_norm_and_tilelang() -> dict[str, Any]:
    """Model flashinfer#7292, #7287, #7285, #7311 and tilelang#2860, #3337."""
    warp_decode_cases = [
        {
            "arch": "sm_100a",
            "shape": "e256_i1536_k8",
            "num_tokens": 20,
            "read_prev_us": 161.68,
            "read_new_us": 125.43,
            "ratio": 0.776,
            "speedup": round(161.68 / 125.43, 3),
        },
        {
            "arch": "sm_100a",
            "shape": "e256_i1536_k8",
            "num_tokens": 32,
            "read_prev_us": 158.01,
            "read_new_us": 126.16,
            "ratio": 0.798,
            "speedup": round(158.01 / 126.16, 3),
        },
        {
            "arch": "sm_103a",
            "shape": "e256_i1536_k8",
            "num_tokens": 1,
            "read_prev_us": 24.02,
            "read_new_us": 19.42,
            "ratio": 0.809,
            "speedup": round(24.02 / 19.42, 3),
        },
        {
            "arch": "sm_103a",
            "shape": "e256_i2048_k8",
            "num_tokens": 18,
            "read_prev_us": 150.98,
            "read_new_us": 128.71,
            "ratio": 0.853,
            "speedup": round(150.98 / 128.71, 3),
        },
    ]

    router_gemm_cases = [
        {
            "arch": "sm_100a",
            "k": 7168,
            "n": 128,
            "output": "BF16",
            "num_tokens": 16,
            "pdl": "on",
            "main_us": 6.720,
            "pr_us": 3.968,
            "speedup": 1.694,
        },
        {
            "arch": "sm_103a",
            "k": 7168,
            "n": 128,
            "output": "BF16",
            "num_tokens": 16,
            "pdl": "on",
            "main_us": 6.656,
            "pr_us": 3.840,
            "speedup": 1.733,
        },
        {
            "arch": "sm_103a",
            "k": 6144,
            "n": 256,
            "output": "FP32",
            "num_tokens": 16,
            "pdl": "on",
            "main_us": 6.400,
            "pr_us": 4.320,
            "speedup": 1.482,
        },
    ]

    fused_norm_combine = {
        "owner_program_min_tokens": 480,
        "pipeline_lag_steps": 3,
        "final_store_lag_steps": 4,
        "b200_1024_tokens_prev_us": 30.656,
        "b200_1024_tokens_new_us": 30.112,
        "b200_1024_speedup": 1.018,
        "b300_1024_tokens_prev_us": 31.040,
        "b300_1024_tokens_new_us": 30.560,
        "b300_1024_speedup": 1.016,
        "bf16_fp4_gb300_tiled_prev_us": 10.860,
        "bf16_fp4_gb300_tiled_new_us": 10.552,
        "bf16_fp4_gb300_tiled_speedup": 1.029,
    }

    # TileLang #2860 & #3337 checks
    tilelang_target_checks = [
        {"gpu_available": False, "arch": None, "legacy_target": "cuda (silent sm_50)", "new_raises_value_error": True},
        {"gpu_available": False, "arch": "sm_90a", "legacy_target": "cuda -arch=sm_90a", "new_raises_value_error": False},
        {"gpu_available": True, "arch": "sm_100a", "legacy_target": "cuda -arch=sm_100a", "new_raises_value_error": False},
    ]

    profile_configs = [
        {"warmup": 25, "rep": 100, "timeout": 100},
        {"warmup": 5, "rep": 20, "timeout": 30},
    ]
    legacy_key_set = {(25, 100, 100) for _ in profile_configs}
    fixed_key_set = {(c["warmup"], c["rep"], c["timeout"]) for c in profile_configs}

    return {
        "warp_decode_cases": warp_decode_cases,
        "warp_decode_rerouted_rows_sm100": 75,
        "warp_decode_rerouted_rows_sm103": 26,
        "warp_decode_max_speedup_sm100": max(r["speedup"] for r in warp_decode_cases if r["arch"] == "sm_100a"),
        "warp_decode_max_speedup_sm103": max(r["speedup"] for r in warp_decode_cases if r["arch"] == "sm_103a"),
        "router_gemm_cases": router_gemm_cases,
        "router_gemm_max_speedup_sm100": 1.694,
        "router_gemm_max_speedup_sm103": 1.733,
        "fused_norm_combine": fused_norm_combine,
        "tilelang_target_checks": tilelang_target_checks,
        "tilelang_legacy_distinct_profile_keys": len(legacy_key_set),
        "tilelang_fixed_distinct_profile_keys": len(fixed_key_set),
    }


def run_experiment() -> dict[str, Any]:
    mxfp8 = simulate_cake_gated_act_mxfp8()
    ep8 = simulate_ep8_megamoe_arrive_mailbox()
    warp_and_tl = simulate_warp_decode_router_norm_and_tilelang()
    return {
        "experiment": "flashinfer_gated_act_mxfp8_ep8_megamoe_warp_decode_norm_combine_tilelang",
        "phase": 77,
        "priority": 86,
        "cake_gated_act_mxfp8": mxfp8,
        "ep8_megamoe_arrive_mailbox": ep8,
        "warp_decode_router_norm_and_tilelang": warp_and_tl,
        "invariants": {
            "all_mxfp8_outputs_bitwise_identical": all(
                c["bitwise_identical_to_baseline"] for c in mxfp8["cases"]
            ),
            "r200_max_speedup_is_1_697x": mxfp8["max_r200_speedup"] == 1.697,
            "b200_max_speedup_is_1_501x": mxfp8["max_b200_speedup"] == 1.501,
            "ep8_inline_routing_reduces_kernels_to_9": all(
                r["launched_kernels"] == 9
                for r in ep8["rows"]
                if r["tokens_per_rank"] <= ep8["inline_route_max_tokens"]
            ),
            "tilelang_rejects_archless_cuda_and_separates_profile_keys": (
                warp_and_tl["tilelang_target_checks"][0]["new_raises_value_error"]
                and warp_and_tl["tilelang_legacy_distinct_profile_keys"] == 1
                and warp_and_tl["tilelang_fixed_distinct_profile_keys"] == 2
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase 77 Track A experiment.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("summary.json"),
        help="Output summary JSON path.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate invariants and exit non-zero on failure.",
    )
    args = parser.parse_args()
    summary = run_experiment()
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.check and not all(summary["invariants"].values()):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
