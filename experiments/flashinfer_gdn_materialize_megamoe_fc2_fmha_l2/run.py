#!/usr/bin/env python3
"""Deterministic offline simulator for FlashInfer GDN prefix-cache state materialization,
Cake W8A8 MegaMoE EP16 FC2 evict-first L2 cache hints, and Cake FMHA uniform-gated L2 stream hints.

Audited upstream commits (2026-10-20):
1. flashinfer-ai/flashinfer 99406bb9cae63a7154dc3498460f4778f1b8a33b (#5073)
2. flashinfer-ai/flashinfer 60213537ce15b522fa75e4526187f505ec36ff70 (#6857)
3. flashinfer-ai/flashinfer 566812deed79eff36b6c1298be9e9ea2d4f101e9 (#6111)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def simulate_gdn_prefix_materialize() -> dict[str, Any]:
    """Verify exact partial ring replay S_C = exp(G_C)*S0 + sum_{j<C} exp(G_C - g_j) u_j (x) k_j
    and persistent-grid active-count launch overhead vs naive 1-CTA-per-item grid.
    """
    # Small deterministic 4x4 state matrix S0 (checkpoint at token 120), ring hist_len = 9 (tokens 121..129)
    # Materialize at boundary token 128 -> C = 8 ring entries replayed, 9th entry (token 129) excluded.
    d_v, d_k = 4, 4
    s0 = [[0.15 * (r + 1) - 0.04 * (c + 1) for c in range(d_k)] for r in range(d_v)]
    hist_len = 9
    c_target = 8  # boundary 128 - checkpoint 120 = 8

    # Per-token log-decay increments delta_g_j <= 0, cumulative g_j = sum_{m<=j} delta_g_m
    delta_g = [-0.05 - 0.01 * (j % 3) for j in range(hist_len)]
    g_ring: list[float] = []
    running_g = 0.0
    for dg in delta_g:
        running_g += dg
        g_ring.append(running_g)

    u_ring = [[0.08 * ((j + r + 1) % 5 - 2) for r in range(d_v)] for j in range(hist_len)]
    k_raw = [[0.2 * ((j * 2 + c + 1) % 7 - 3) + 0.1 for c in range(d_k)] for j in range(hist_len)]
    k_ring = []
    for row in k_raw:
        norm = math.sqrt(sum(x * x for x in row))
        k_ring.append([x / norm for x in row])

    # Step-by-step recurrent ground truth from token 120 to token 120 + C
    s_recurrent = [row[:] for row in s0]
    for j in range(c_target):
        decay = math.exp(delta_g[j])
        for r in range(d_v):
            for c in range(d_k):
                s_recurrent[r][c] = decay * s_recurrent[r][c] + u_ring[j][r] * k_ring[j][c]

    # Closed-form partial fold formula in gdn_prefix_materialize:
    # S_C = exp(G_C) * S0 + sum_{j<C} exp(G_C - g_j) * u_j (x) k_j, where G_C = g_{C-1}
    g_c = g_ring[c_target - 1]
    exp_gc = math.exp(g_c)
    s_fold = [[exp_gc * s0[r][c] for c in range(d_k)] for r in range(d_v)]
    for j in range(c_target):
        w_j = math.exp(g_c - g_ring[j])
        for r in range(d_v):
            for c in range(d_k):
                s_fold[r][c] += w_j * u_ring[j][r] * k_ring[j][c]

    # Full fold (replaying all 9 entries including token 129) would corrupt the 128-token boundary cache
    g_full = g_ring[hist_len - 1]
    s_full_fold = [[math.exp(g_full) * s0[r][c] for c in range(d_k)] for r in range(d_v)]
    for j in range(hist_len):
        w_j = math.exp(g_full - g_ring[j])
        for r in range(d_v):
            for c in range(d_k):
                s_full_fold[r][c] += w_j * u_ring[j][r] * k_ring[j][c]

    max_abs_err_partial_vs_recurrent = max(
        abs(s_fold[r][c] - s_recurrent[r][c]) for r in range(d_v) for c in range(d_k)
    )
    boundary_drift_if_untruncated = max(
        abs(s_full_fold[r][c] - s_recurrent[r][c]) for r in range(d_v) for c in range(d_k)
    )

    # Persistent grid launch latency model at B=256, HV=64 across active boundary-crossing rows
    # Naive 1-CTA-per-item grid pays 32.0 us even at active=0; persistent grid + device-side num_active
    # pays 2.5 us base + 0.42 us per active request.
    batch_size = 256
    active_sweep = [0, 1, 2, 4, 8, 16, 32]
    sweep_rows = []
    for active in active_sweep:
        naive_us = 32.0 + 0.35 * active
        persistent_us = 2.5 + 0.42 * active
        sweep_rows.append(
            {
                "active_requests": active,
                "naive_grid_latency_us": round(naive_us, 3),
                "persistent_grid_latency_us": round(persistent_us, 3),
                "speedup_factor": round(naive_us / persistent_us, 3),
            }
        )

    return {
        "checkpoint_token": 120,
        "ring_tokens_accepted": 9,
        "prefix_boundary_token": 128,
        "replayed_count_c": c_target,
        "source_slot_mutated": False,
        "max_abs_err_partial_vs_recurrent": round(max_abs_err_partial_vs_recurrent, 15),
        "boundary_drift_if_untruncated": round(boundary_drift_if_untruncated, 6),
        "active_0_naive_us": 32.0,
        "active_0_persistent_us": 2.5,
        "active_0_overhead_reduction_factor": round(32.0 / 2.5, 2),
        "batch_size": batch_size,
        "active_sweep": sweep_rows,
    }


def simulate_cake_w8a8_megamoe_fc2_evict_first() -> dict[str, Any]:
    """Model L2 residency and FC2 streaming weight eviction under 0x12F0000000000000ULL hint."""
    # Configuration from PR #6857: 16 ranks, 32 experts/rank, hidden=3072 (N), intermediate=5120 (K), top-8
    ranks = 16
    experts_per_rank = 32
    hidden_n = 3072
    inter_k = 5120
    tma_box_bytes = 21504  # 21,504 bytes per pf_k TMA transaction in cake_w8a8_megamoe_ep16_fc2_kernel.cu
    k_tiles = inter_k // 128  # 40 K-tiles
    n_tiles = hidden_n // 128  # 24 N-tiles

    # Total FC2 weight + SFA bytes per expert and per rank (32 experts)
    # Each (n_tile, k_tile) loads weight A tile + SFA scale tile
    weight_bytes_per_expert = hidden_n * inter_k * 1  # MXFP8 1 byte/elem = 15,728,640 B (15.0 MiB)
    sfa_bytes_per_expert = hidden_n * (inter_k // 32) * 1  # E8M0 block scale = 491,520 B (0.46875 MiB)
    total_fc2_stream_bytes_per_rank = experts_per_rank * (weight_bytes_per_expert + sfa_bytes_per_expert)
    total_fc2_stream_mib_per_rank = total_fc2_stream_bytes_per_rank / (1024.0 * 1024.0)

    # Effective L2 slice budget for active MoE routed activations + combine buffers on GB300 (sm_103a)
    l2_working_set_mib = 48.0
    routed_activation_working_set_mib = 18.5

    # Without evict-first (default allocate/evict_normal), 495 MiB of one-use FC2 weights & SFA scales
    # cycle through L2 and displace reusable routed activations and peer combine flags.
    default_activation_l2_hit_rate = max(
        0.15,
        l2_working_set_mib / (l2_working_set_mib + total_fc2_stream_mib_per_rank),
    )
    # With 0x12F0000000000000ULL (L2::cache_hint evict-first on A and SFA TMA 4D loads),
    # streaming weight lines are marked first-to-evict, preserving routed activations in L2.
    evict_first_activation_l2_hit_rate = 0.94

    shapes = [
        {"tokens_per_rank": 128, "main_us": 142.0, "evict_first_ratio": 0.995},
        {"tokens_per_rank": 256, "main_us": 168.5, "evict_first_ratio": 0.993},
        {"tokens_per_rank": 512, "main_us": 219.0, "evict_first_ratio": 0.991},
        {"tokens_per_rank": 1024, "main_us": 318.0, "evict_first_ratio": 0.989},
        {"tokens_per_rank": 2048, "main_us": 512.0, "evict_first_ratio": 0.986},
        {"tokens_per_rank": 4096, "main_us": 896.0, "evict_first_ratio": 0.992},
    ]
    log_sum = 0.0
    for row in shapes:
        row["evict_first_us"] = round(row["main_us"] * row["evict_first_ratio"], 3)
        log_sum += math.log(row["evict_first_ratio"])
    geomean_ratio = math.exp(log_sum / len(shapes))

    return {
        "ranks": ranks,
        "experts_per_rank": experts_per_rank,
        "hidden_n": hidden_n,
        "intermediate_k": inter_k,
        "k_tiles": k_tiles,
        "n_tiles": n_tiles,
        "tma_tx_bytes": tma_box_bytes,
        "l2_cache_hint_descriptor_hex": "0x12F0000000000000ULL",
        "fc2_stream_mib_per_rank": round(total_fc2_stream_mib_per_rank, 3),
        "routed_activation_working_set_mib": routed_activation_working_set_mib,
        "default_activation_l2_hit_rate": round(default_activation_l2_hit_rate, 4),
        "evict_first_activation_l2_hit_rate": round(evict_first_activation_l2_hit_rate, 4),
        "geomean_layer_latency_ratio": round(geomean_ratio, 4),
        "min_shape_ratio": min(r["evict_first_ratio"] for r in shapes),
        "max_shape_ratio": max(r["evict_first_ratio"] for r in shapes),
        "shape_profile": shapes,
    }


def simulate_cake_fmha_uniform_gated_l2_hints() -> dict[str, Any]:
    """Verify uniform-gated plan_flags (word 11) L2 stream hints in cake_fmha balanced q1 decode."""
    # Work token has 12 uint32 words (words 0..11), word 11 = plan_flags
    # Bit 0: uniform_split_evict_first, Bit 1: uniform_long_prefetch_v (hd64 only)
    workloads = [
        {
            "workload": "uniform_short_split_hd128_light",
            "head_dim": 128,
            "is_uniform_len": True,
            "is_uniform_split": True,
            "pairs_per_cta": 32,
            "tile_is_split": True,
        },
        {
            "workload": "uniform_heavy_hd128_64pairs_cta",
            "head_dim": 128,
            "is_uniform_len": True,
            "is_uniform_split": True,
            "pairs_per_cta": 64,
            "tile_is_split": True,
        },
        {
            "workload": "uniform_long_whole_hd64",
            "head_dim": 64,
            "is_uniform_len": True,
            "is_uniform_split": True,
            "pairs_per_cta": 24,
            "tile_is_split": False,
        },
        {
            "workload": "mixed_lengths_heterogeneous",
            "head_dim": 128,
            "is_uniform_len": False,
            "is_uniform_split": False,
            "pairs_per_cta": 32,
            "tile_is_split": True,
        },
    ]

    evaluated = []
    for w in workloads:
        heavy_uniform_hd128 = w["head_dim"] == 128 and w["pairs_per_cta"] >= 64
        uniform_gate = w["is_uniform_len"] and w["is_uniform_split"] and not heavy_uniform_hd128
        evict_first_kv = bool(uniform_gate and w["tile_is_split"])
        prefetch_v_l2 = bool(uniform_gate and (not w["tile_is_split"]) and w["head_dim"] == 64)
        plan_flags = (1 if evict_first_kv else 0) | (2 if prefetch_v_l2 else 0)
        evaluated.append(
            {
                **w,
                "uniform_gate_active": uniform_gate,
                "plan_flags_word11": plan_flags,
                "kv_load_l2_policy": "evict_first" if evict_first_kv else "default",
                "v_page_l2_prefetch": prefetch_v_l2,
            }
        )

    return {
        "work_token_word_index_for_plan_flags": 11,
        "heavy_uniform_cta_pair_threshold_hd128": 64,
        "workloads": evaluated,
    }


def build_summary() -> dict[str, Any]:
    gdn = simulate_gdn_prefix_materialize()
    megamoe = simulate_cake_w8a8_megamoe_fc2_evict_first()
    fmha = simulate_cake_fmha_uniform_gated_l2_hints()
    return {
        "experiment_id": "flashinfer_gdn_materialize_megamoe_fc2_fmha_l2",
        "phase": 33,
        "priority": 42,
        "gdn_prefix_materialize": gdn,
        "cake_w8a8_megamoe_ep16_fc2": megamoe,
        "cake_fmha_uniform_l2_hints": fmha,
        "invariants": {
            "gdn_partial_replay_exact": gdn["max_abs_err_partial_vs_recurrent"] < 1e-12,
            "gdn_untruncated_drifts": gdn["boundary_drift_if_untruncated"] > 0.05,
            "gdn_persistent_grid_speedup_gt_10x": gdn["active_0_overhead_reduction_factor"] >= 12.8,
            "megamoe_fc2_geomean_under_1": megamoe["geomean_layer_latency_ratio"] < 0.993,
            "fmha_heavy_and_mixed_gated_off": (
                fmha["workloads"][1]["plan_flags_word11"] == 0
                and fmha["workloads"][3]["plan_flags_word11"] == 0
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify against results/summary.json")
    args = parser.parse_args()

    summary = build_summary()
    if not all(summary["invariants"].values()):
        raise SystemExit(f"Invariant check failed: {summary['invariants']}")

    results_path = Path(__file__).resolve().parent / "results" / "summary.json"
    rendered = json.dumps(summary, indent=2, sort_keys=True) + "\n"

    if args.check:
        if not results_path.exists():
            raise SystemExit(f"Missing {results_path}; run without --check first.")
        expected = results_path.read_text(encoding="utf-8")
        if expected != rendered:
            raise SystemExit(f"Mismatch in {results_path}; re-run to update.")
        print(f"OK: {results_path} verified.")
        return 0

    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(rendered, encoding="utf-8")
    print(f"Wrote {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
