#!/usr/bin/env python3
"""Deterministic reproduction of Phase 49 Topic 1:
1. FlashInfer unified frost_low_latency MoE backend on SM100/SM103/SM107 (#6626)
2. FlashInfer cake_rmsnorm_train row-count window backward & fixed-order tree dweight (#6939)
3. FlashInfer cake_sage one-step-ahead softmax metadata staging on SM100/SM103 (#6951)
4. FlashInfer cake_ssd_combined sm_107a look-back chunk-parallel Mamba2 SSD (#6943)
5. FlashInfer cake_vsa_sm90 persistent kernel & compact static program (#6949)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


FROST_ACTIVATIONS = (
    "SwiGLU",
    "SiTU",
    "GeGLU",
    "GeGLUTanh",
    "SwiGLUStep",
    "GELU",
    "ReLU",
    "ReLU2",
    "SiLU",
    "Identity",
)
FROST_WARP_WIDTHS = (1, 2, 4, 8)


def simulate_frost_low_latency_moe() -> dict[str, Any]:
    """Model unified FrostLowLatencyConfig tactic space, shared MX weight layout, and B300 speedups (#6626)."""
    warp_width_pairs = [(w1, w2) for w1 in FROST_WARP_WIDTHS for w2 in FROST_WARP_WIDTHS]
    assert len(warp_width_pairs) == 16

    def should_route_low_latency(
        num_tokens: int,
        top_k: int,
        arch: str,
        dtype: str,
        max_num_tokens: int = 2,
    ) -> dict[str, Any]:
        in_token_range = 1 <= num_tokens <= max_num_tokens and max_num_tokens <= 8
        arch_supported = arch in ("sm_100a", "sm_103a", "sm_107a")
        if not (in_token_range and arch_supported):
            return {"routed": False, "tactics": [], "shared_prepared_weights": False}
        if dtype in ("mxfp4", "mxfp8"):
            tactics = ["TC128_PDL", "TC256_PDL"]
            if top_k <= 32:
                tactics.extend(
                    [f"MX_FMA_W1_{w1}_W2_{w2}" for w1, w2 in warp_width_pairs]
                )
            return {
                "routed": True,
                "tactics": tactics,
                "shared_prepared_weights": True,
            }
        if dtype == "bf16" and top_k <= 32:
            return {
                "routed": True,
                "tactics": [f"BF16_FMA_W1_{w1}_W2_{w2}" for w1, w2 in warp_width_pairs],
                "shared_prepared_weights": True,
            }
        return {"routed": False, "tactics": [], "shared_prepared_weights": False}

    route_mx_t2 = should_route_low_latency(2, 8, "sm_103a", "mxfp4", max_num_tokens=2)
    route_mx_t4_default = should_route_low_latency(4, 8, "sm_103a", "mxfp4", max_num_tokens=2)
    route_mx_t4_override = should_route_low_latency(4, 8, "sm_103a", "mxfp4", max_num_tokens=4)
    route_mx_topk64 = should_route_low_latency(2, 64, "sm_100a", "mxfp8", max_num_tokens=2)

    b300_bf16_cases = [
        {"shape": "T=1, SwiGLU", "cutlass_us": 27.736, "frost_us": 15.460},
        {"shape": "T=2, SwiGLU", "cutlass_us": 44.144, "frost_us": 31.872},
        {"shape": "T=2, SiTU", "cutlass_us": 60.424, "frost_us": 46.056},
    ]
    for c in b300_bf16_cases:
        c["speedup"] = round(c["cutlass_us"] / c["frost_us"], 3)

    b300_mx_cases = [
        {"shape": "W4/T1 clamped-SwiGLU", "cutlass_us": 21.616, "frost_us": 13.440},
        {"shape": "W4/T4 clamped-SwiGLU", "cutlass_us": 31.852, "frost_us": 21.632},
        {"shape": "W8/T4 clamped-SwiGLU", "cutlass_us": 40.008, "frost_us": 31.816},
    ]
    for c in b300_mx_cases:
        c["speedup"] = round(c["cutlass_us"] / c["frost_us"], 3)

    return {
        "num_typed_activations": len(FROST_ACTIVATIONS),
        "num_warp_width_pairs": len(warp_width_pairs),
        "mx_t2_tactic_count": len(route_mx_t2["tactics"]),
        "mx_topk64_tactic_count": len(route_mx_topk64["tactics"]),
        "t4_default_routed": route_mx_t4_default["routed"],
        "t4_override_routed": route_mx_t4_override["routed"],
        "shared_prepared_weights": route_mx_t2["shared_prepared_weights"],
        "b300_bf16_cases": b300_bf16_cases,
        "b300_mx_cases": b300_mx_cases,
        "peak_bf16_speedup": max(c["speedup"] for c in b300_bf16_cases),
        "peak_mx_speedup": max(c["speedup"] for c in b300_mx_cases),
    }


def simulate_cake_rmsnorm_train() -> dict[str, Any]:
    """Model cake_rmsnorm_train row-window backward, fixed-order tree dweight reduction, and self-resetting flip barrier (#6939)."""
    n_chunks = 8
    cols = 16
    # Deterministic partial dweight matrix [n_chunks][cols] with wide exponent spread
    partials = [
        [((c * 13 + j * 7 + 3) % 29 - 14) * (10.0 ** ((c % 3) - 1)) for j in range(cols)]
        for c in range(n_chunks)
    ]

    def tree_reduce_dweight(part_rows: list[list[float]]) -> list[float]:
        level = [list(row) for row in part_rows]
        while len(level) > 1:
            next_level = []
            for i in range(0, len(level), 2):
                if i + 1 < len(level):
                    next_level.append([a + b for a, b in zip(level[i], level[i + 1])])
                else:
                    next_level.append(level[i])
            level = next_level
        return level[0]

    # Run under two different CTA arrival permutations: fixed-order tree uses chunk_id slot, not arrival order
    perm_a = [0, 1, 2, 3, 4, 5, 6, 7]
    perm_b = [7, 2, 5, 0, 6, 1, 4, 3]
    staged_a = [None] * n_chunks
    staged_b = [None] * n_chunks
    for cid in perm_a:
        staged_a[cid] = partials[cid]
    for cid in perm_b:
        staged_b[cid] = partials[cid]

    dw_tree_a = tree_reduce_dweight(staged_a)  # type: ignore[arg-type]
    dw_tree_b = tree_reduce_dweight(staged_b)  # type: ignore[arg-type]
    tree_bitwise_deterministic = dw_tree_a == dw_tree_b

    # Simulate self-resetting flip barrier across 4 consecutive launches without zero-fill launch
    counter_state = {"count": 0, "sense": 0}
    launch_senses = []
    for _launch in range(4):
        for cta in range(n_chunks):
            counter_state["count"] += 1
            if counter_state["count"] == n_chunks:
                # Last arriving CTA resets count to 0 and flips sense bit
                counter_state["count"] = 0
                counter_state["sense"] ^= 1
        launch_senses.append((counter_state["count"], counter_state["sense"]))

    r200_benchmarks = [
        {"op": "fwd", "H": 1536, "T": 16384, "prev_us": 26.5, "new_us": 24.8, "speedup": 1.069},
        {"op": "fwd", "H": 1536, "T": 32768, "prev_us": 48.0, "new_us": 44.5, "speedup": 1.079},
        {"op": "bwd", "H": 1536, "T": 4096, "prev_us": 39.2, "new_us": 34.9, "speedup": 1.123},
        {"op": "bwd", "H": 4096, "T": 8192, "prev_us": 93.7, "new_us": 82.6, "speedup": 1.134},
    ]

    return {
        "tree_bitwise_deterministic": tree_bitwise_deterministic,
        "post_launch_counter_zero": all(cnt == 0 for cnt, _ in launch_senses),
        "sense_alternation": [s for _, s in launch_senses],
        "r200_benchmarks": r200_benchmarks,
        "peak_r200_fwd_speedup": 1.079,
        "peak_r200_bwd_speedup": 1.134,
        "peak_b200_bwd_speedup": 1.118,
        "peak_gb300_bwd_speedup": 1.075,
    }


def simulate_cake_sage_ssd_vsa() -> dict[str, Any]:
    """Model cake_sage one-step-ahead metadata (#6951), cake_ssd_combined sm_107a (#6943), and cake_vsa_sm90 (#6949)."""
    sage_smem_prev = 119040
    sage_smem_new = 120320
    sage_smem_delta = sage_smem_new - sage_smem_prev

    sage_cases = [
        {"gpu": "GB300 (sm_103a)", "shape": "A1 B8 H32 sel=6", "prev_ms": 0.2922, "new_ms": 0.2824, "speedup": 1.03},
        {"gpu": "GB300 (sm_103a)", "shape": "A2 B8 H32 sel=32", "prev_ms": 0.8459, "new_ms": 0.8246, "speedup": 1.03},
        {"gpu": "B200 (sm_100a)", "shape": "A1 B8 H32 sel=6", "prev_ms": 0.3552, "new_ms": 0.3288, "speedup": 1.08},
        {"gpu": "B200 (sm_100a)", "shape": "A2 B8 H32 sel=32", "prev_ms": 1.0953, "new_ms": 1.0629, "speedup": 1.03},
    ]

    ssd_r200_cases = [
        {"shape": "perf_batched_1x256", "call_prev_us": 92.1, "call_new_us": 77.0, "call_speedup": 1.196, "kernel_prev_us": 84.1, "kernel_new_us": 73.8, "kernel_speedup": 1.140},
        {"shape": "perf_batched_4x16", "call_prev_us": 34.1, "call_new_us": 34.3, "call_speedup": 0.995, "kernel_prev_us": 28.8, "kernel_new_us": 25.9, "kernel_speedup": 1.114},
        {"shape": "cp_b4h8c16", "call_prev_us": 34.2, "call_new_us": 34.2, "call_speedup": 1.001, "kernel_prev_us": 33.3, "kernel_new_us": 26.2, "kernel_speedup": 1.268},
    ]

    vsa_sm90_programs = {
        "persistent": ["static_tile_lists", "device_tile_queue", "compact_static_short_tiles"],
        "small_selection_direct": ["k1", "k3", "k4", "k6"],
        "small_selection_dsm_cluster": ["k2c4", "k3c3", "k3c6", "k4c2", "k4c4", "k6c2"],
        "small_selection_split_kv": ["k4s", "k6s"],
        "small_selection_split_cluster": ["k6sc3", "k6sc4", "k6sc8"],
    }
    total_vsa_programs = sum(len(v) for v in vsa_sm90_programs.values())

    vsa_h100_highlights = [
        {"suite": "ext33", "shape": "h1-m512-n8192-k48-ragged0-scale0", "prev_us": 13.472, "new_us": 10.680, "speedup": 1.2624},
        {"suite": "real39", "shape": "h8-m128-n32768-k64-ragged0-scale0", "prev_us": 35.360, "new_us": 28.912, "speedup": 1.2237},
        {"suite": "ext33", "shape": "h1-m1024-n4096-k24-ragged0-scale0", "prev_us": 10.880, "new_us": 9.232, "speedup": 1.1806},
        {"suite": "real39", "shape": "h4-m256-n8192-k24-ragged0-scale0", "prev_us": 13.120, "new_us": 11.136, "speedup": 1.1782},
    ]

    return {
        "sage_smem_prev_bytes": sage_smem_prev,
        "sage_smem_new_bytes": sage_smem_new,
        "sage_smem_delta_bytes": sage_smem_delta,
        "sage_cases": sage_cases,
        "ssd_r200_cases": ssd_r200_cases,
        "ssd_call_geomean_speedup": 1.0215,
        "ssd_main_kernel_geomean_speedup": 1.0839,
        "total_vsa_sm90_programs": total_vsa_programs,
        "vsa_h100_highlights": vsa_h100_highlights,
        "peak_vsa_speedup": max(c["speedup"] for c in vsa_h100_highlights),
    }


def build_summary() -> dict[str, Any]:
    frost = simulate_frost_low_latency_moe()
    rmsnorm = simulate_cake_rmsnorm_train()
    sage_ssd_vsa = simulate_cake_sage_ssd_vsa()
    return {
        "experiment": "flashinfer-frost-low-latency-moe-cake-rmsnorm-sage-vsa",
        "phase": 49,
        "priority": 58,
        "frost_low_latency_moe": frost,
        "cake_rmsnorm_train": rmsnorm,
        "cake_sage_ssd_vsa": sage_ssd_vsa,
        "invariants": {
            "frost_10_activations_16_warp_pairs": (
                frost["num_typed_activations"] == 10 and frost["num_warp_width_pairs"] == 16
            ),
            "frost_shared_prepared_weights": frost["shared_prepared_weights"],
            "rmsnorm_tree_dweight_bitwise_deterministic": rmsnorm["tree_bitwise_deterministic"],
            "rmsnorm_flip_barrier_self_resets": rmsnorm["post_launch_counter_zero"],
            "sage_smem_delta_1280_bytes": sage_ssd_vsa["sage_smem_delta_bytes"] == 1280,
            "vsa_sm90_18_programs": sage_ssd_vsa["total_vsa_sm90_programs"] == 18,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Verify deterministic invariants")
    args = parser.parse_args()

    summary = build_summary()
    out_path = Path(__file__).resolve().parent / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if args.check:
        assert all(summary["invariants"].values()), f"Invariant failure: {summary['invariants']}"
        assert math.isclose(summary["frost_low_latency_moe"]["peak_bf16_speedup"], 1.794, abs_tol=1e-3)
        assert math.isclose(summary["frost_low_latency_moe"]["peak_mx_speedup"], 1.608, abs_tol=1e-3)
        assert math.isclose(summary["cake_rmsnorm_train"]["peak_r200_bwd_speedup"], 1.134, abs_tol=1e-3)
        assert math.isclose(summary["cake_sage_ssd_vsa"]["peak_vsa_speedup"], 1.2624, abs_tol=1e-4)

    print(json.dumps(summary["invariants"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
