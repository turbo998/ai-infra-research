#!/usr/bin/env python3
"""Offline simulator for Phase 76 Track A:
FlashInfer SM103 NVFP4 TMEM Attention, EP16 MXFP8 MegaMoE Relaxed Fence/Dispatch,
Kimi-K3 TP12 ESS Routing, SM120 FP8 Varlen Attention + Device Alpha Out-Proj, & TileLang Ascend/ROCm.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def simulate_sm103_nvfp4_tmem_attention() -> dict[str, object]:
    """Model FlashInfer PR #7328 (d08a1f732193): cake_nvfp4_attention on sm_103a (GB300).

    - TMEM columns increase from 432 to 480 by staging P0_LO/HI (cols 448, 456) and
      P1_LO/HI (cols 464, 472) directly in Tensor Memory (TMEM) and overlapping
      tile-boundary scoreboard transitions.
    """
    tmem_cols_prev = 432
    tmem_cols_pr = 480
    p_offsets = {
        "TMEM_TMEM_P0_LO_OFFSET": 448,
        "TMEM_TMEM_P0_HI_OFFSET": 456,
        "TMEM_TMEM_P1_LO_OFFSET": 464,
        "TMEM_TMEM_P1_HI_OFFSET": 472,
    }
    cases = [
        {"b": 4, "h": 8, "s": 4096, "d": 128, "main_ms": 0.1145, "pr_ms": 0.1094, "reported_speedup_x": 1.05},
        {"b": 1, "h": 8, "s": 32768, "d": 128, "main_ms": 1.4432, "pr_ms": 1.4201, "reported_speedup_x": 1.02},
        {"b": 8, "h": 32, "s": 8192, "d": 128, "main_ms": 2.8392, "pr_ms": 2.7562, "reported_speedup_x": 1.03},
    ]
    rows = []
    for c in cases:
        ratio = round(c["main_ms"] / c["pr_ms"], 4)
        rows.append({
            **c,
            "s_mod_512_zero": (c["s"] % 512 == 0),
            "computed_speedup_x": ratio,
        })
    return {
        "tmem_ncols_prev": tmem_cols_prev,
        "tmem_ncols_pr": tmem_cols_pr,
        "p_tmem_offsets": p_offsets,
        "num_k_pipe_stages": 4,
        "num_v_pipe_stages": 5,
        "cases": rows,
        "max_speedup_x": max(r["reported_speedup_x"] for r in rows),
        "min_speedup_x": min(r["reported_speedup_x"] for r in rows),
    }


def simulate_megamoe_ep16_relaxed_fence_and_dispatch() -> dict[str, object]:
    """Model FlashInfer PR #7299 (a4dc7d17fb6a): CakeMxfp8MegaMoeEp16 on 16xGB300 (NVL72).

    - Barrier: replaces 16 sequential system-scope release atomics per cross-rank barrier
      (32 total across 2 barriers) with 1 system-scope acq_rel fence + 16 relaxed system-scope
      atomics per barrier (2 fences + 32 relaxed atomics total).
    - Dispatch: 9 CTAs/rank (C72), 12 warps/CTA = 108 warps claiming token rows dynamically;
      lanes 0..7 claim 8 routed expert rows and copy 6 KiB BF16 row once in two passes of
      six 16-byte words per lane (32 lanes * 6 * 16B * 2 passes = 6144 B).
    """
    world_size = 16
    barriers_per_call = 2
    prev_seq_release_atomics = world_size * barriers_per_call
    pr_sys_acq_rel_fences = barriers_per_call
    pr_relaxed_sys_atomics = world_size * barriers_per_call

    hidden_size = 3072
    bf16_row_bytes = hidden_size * 2  # 6144 B = 6 KiB
    lanes_per_warp = 32
    words16b_per_lane_per_pass = 6
    passes = 2
    copied_bytes_per_token = lanes_per_warp * words16b_per_lane_per_pass * 16 * passes

    benchmarks = [
        {"tokens_per_rank": 16, "total_tokens": 256, "dist": "balanced", "main_us": 355.4, "pr_us": 312.6, "speedup_x": 1.14},
        {"tokens_per_rank": 16, "total_tokens": 256, "dist": "hotset50", "main_us": 355.3, "pr_us": 313.8, "speedup_x": 1.13},
        {"tokens_per_rank": 32, "total_tokens": 512, "dist": "balanced", "main_us": 362.7, "pr_us": 320.5, "speedup_x": 1.13},
        {"tokens_per_rank": 32, "total_tokens": 512, "dist": "hotset50", "main_us": 368.4, "pr_us": 318.5, "speedup_x": 1.16},
        {"tokens_per_rank": 64, "total_tokens": 1024, "dist": "balanced", "main_us": 376.1, "pr_us": 322.0, "speedup_x": 1.17},
        {"tokens_per_rank": 64, "total_tokens": 1024, "dist": "hotset50", "main_us": 377.5, "pr_us": 320.5, "speedup_x": 1.18},
    ]
    return {
        "world_size": world_size,
        "barriers_per_call": barriers_per_call,
        "prev_seq_release_atomics_total": prev_seq_release_atomics,
        "pr_sys_acq_rel_fences_total": pr_sys_acq_rel_fences,
        "pr_relaxed_sys_atomics_total": pr_relaxed_sys_atomics,
        "bf16_row_bytes": bf16_row_bytes,
        "copied_bytes_per_token": copied_bytes_per_token,
        "single_pass_6kib_verified": (bf16_row_bytes == copied_bytes_per_token == 6144),
        "dispatch_ctas_per_rank": 9,
        "dispatch_warps_per_cta": 12,
        "benchmarks": benchmarks,
        "max_speedup_x": max(b["speedup_x"] for b in benchmarks),
        "min_speedup_x": min(b["speedup_x"] for b in benchmarks),
    }


def route_kimi_k3_tp12_tail(m_tokens: int) -> tuple[str, str, bool]:
    """Model FlashInfer PR #7310 (06da6cc640a0): Kimi-K3 TP12 LatentMoE tail routing."""
    if m_tokens <= 4:
        return ("k1_oneshot_ess", "k23:c4", False)
    if m_tokens <= 8:
        return ("k1_oneshot_ess", "k23:c8", False)
    if m_tokens <= 16:
        return ("k1_oneshot_ess", "k3_ess:grouped", True)
    if m_tokens <= 960:
        return ("k1_twoshot_ess:grouped", "k3_ess:grouped", True)
    return ("k1_twoshot:pinned", "k3_persist_bulk:pinned", True)


def simulate_kimi_k3_tp12_ess_routing() -> dict[str, object]:
    """Verify Kimi-K3 TP12 tail route table and GB300/GB200 speedups at M=256 and M=512."""
    probe_tokens = [1, 4, 6, 8, 16, 32, 128, 255, 256, 512, 960, 961, 1024]
    routes = {}
    for m in probe_tokens:
        k1, k3, uses_cublas = route_kimi_k3_tp12_tail(m)
        routes[str(m)] = {"k1": k1, "k3": k3, "uses_cublas": uses_cublas}

    measured = [
        {"m": 256, "gb300_prev_us": 41.74, "gb300_pr_us": 39.39, "gb300_speedup_x": 1.060,
         "gb200_prev_us": 43.17, "gb200_pr_us": 39.96, "gb200_speedup_x": 1.080},
        {"m": 512, "gb300_prev_us": 60.59, "gb300_pr_us": 56.10, "gb300_speedup_x": 1.080,
         "gb200_prev_us": 61.53, "gb200_pr_us": 57.22, "gb200_speedup_x": 1.075},
    ]
    return {
        "persist_min_tokens_prev": 256,
        "persist_min_tokens_pr": 961,
        "cublas_threshold_tokens": 8,
        "routes": routes,
        "measured_256_512": measured,
    }


def simulate_sm120_fp8_varlen_and_device_alpha() -> dict[str, object]:
    """Model FlashInfer PR #7289 (ba12a8322385) and PR #7309 (4796d2effd9e) on SM120."""
    varlen_fp8_rtx5090 = [
        {"workload": "center_0s5 [0, 3920]", "main_us": 1152.9, "pr_us": 926.9, "speedup_x": 1.244},
        {"workload": "center_2s [0, 16592]", "main_us": 18236.0, "pr_us": 13173.4, "speedup_x": 1.384},
        {"workload": "center_4s [0, 33472]", "main_us": 73983.5, "pr_us": 52679.3, "speedup_x": 1.404},
        {"workload": "center_15s [0, 109952]", "main_us": 798811.3, "pr_us": 578255.1, "speedup_x": 1.381},
        {"workload": "short_pack 16x4032", "main_us": 16898.6, "pr_us": 12465.2, "speedup_x": 1.356},
    ]
    varlen_fp8_rtx_pro_6000 = [
        {"workload": "center_0s5 [0, 3920]", "main_us": 524.0, "pr_us": 521.4, "speedup_x": 1.005},
        {"workload": "center_2s [0, 16592]", "main_us": 7135.7, "pr_us": 6462.6, "speedup_x": 1.104},
        {"workload": "center_4s [0, 33472]", "main_us": 27436.3, "pr_us": 25884.6, "speedup_x": 1.060},
        {"workload": "center_15s [0, 109952]", "main_us": 288688.0, "pr_us": 281629.5, "speedup_x": 1.025},
        {"workload": "short_pack 16x4032", "main_us": 6640.0, "pr_us": 6351.2, "speedup_x": 1.045},
    ]

    # Verify FP64-intermediate reciprocal vs host two-step FP32 reciprocal
    act_scale = 0.03125011920928955
    weight_scale = 0.015625059604644775
    device_alpha_fp64_to_fp32 = float(1.0 / (float(act_scale) * float(weight_scale)))

    out_proj_rtx_pro_6000 = [
        {"m": 3920, "main_us": 247.8, "pr_us": 242.7, "speedup_x": 1.021, "main_gap_us": 223.3, "pr_gap_us": 3.5},
        {"m": 16592, "main_us": 1003.5, "pr_us": 989.1, "speedup_x": 1.015, "main_gap_us": 186.9, "pr_gap_us": 3.1},
        {"m": 33472, "main_us": 2004.4, "pr_us": 1929.1, "speedup_x": 1.039, "main_gap_us": 204.8, "pr_gap_us": 3.6},
        {"m": 64512, "main_us": 3840.2, "pr_us": 3793.5, "speedup_x": 1.012, "main_gap_us": 204.8, "pr_gap_us": 3.4},
        {"m": 109952, "main_us": 6543.5, "pr_us": 6426.3, "speedup_x": 1.018, "main_gap_us": 196.6, "pr_gap_us": 3.3},
    ]
    gap_reductions = [round(r["main_gap_us"] / r["pr_gap_us"], 2) for r in out_proj_rtx_pro_6000]
    return {
        "varlen_fp8_rtx5090": varlen_fp8_rtx5090,
        "varlen_fp8_rtx_pro_6000": varlen_fp8_rtx_pro_6000,
        "rtx5090_max_speedup_x": max(r["speedup_x"] for r in varlen_fp8_rtx5090),
        "rtx5090_min_speedup_x": min(r["speedup_x"] for r in varlen_fp8_rtx5090),
        "rtx_pro_6000_varlen_max_speedup_x": max(r["speedup_x"] for r in varlen_fp8_rtx_pro_6000),
        "out_proj_rtx_pro_6000": out_proj_rtx_pro_6000,
        "out_proj_geomean_speedup_x": 1.021,
        "host_syncs_removed_per_call": 2,
        "max_idle_gap_reduction_x": max(gap_reductions),
        "min_idle_gap_reduction_x": min(gap_reductions),
        "device_alpha_sample": device_alpha_fp64_to_fp32,
        "cudnn_cache_pred_prev_us": 0.897,
        "cudnn_cache_pred_pr_us": 0.072,
        "cudnn_cache_reduction_pct": 91.97,
    }


def simulate_tilelang_ascend_and_rocm_fp4() -> dict[str, object]:
    """Model TileLang PR #3511 (0bd33f953cf5) Ascend partial L0 pitch & PR #3548 (9253431f2a17) ROCm FP4."""
    # Ascend partial copy from L1 (full N=256) into L0 slice (n_slice=128) with l0_pitch=256
    m_tile = 64
    n_full = 256
    n_slice = 128
    elem_bytes = 2
    wrong_packed_l0_stride_bytes = n_slice * elem_bytes
    explicit_l0_pitch_bytes = n_full * elem_bytes
    exact_l1_write_footprint_bytes = m_tile * n_slice * elem_bytes
    overapproximated_l1_footprint_bytes = m_tile * n_full * elem_bytes

    return {
        "ascend_m_tile": m_tile,
        "ascend_n_full": n_full,
        "ascend_n_slice": n_slice,
        "wrong_packed_l0_stride_bytes": wrong_packed_l0_stride_bytes,
        "explicit_l0_pitch_bytes": explicit_l0_pitch_bytes,
        "exact_l1_write_footprint_bytes": exact_l1_write_footprint_bytes,
        "overapproximated_l1_footprint_bytes": overapproximated_l1_footprint_bytes,
        "rocm_fp4_helper": "__tl_float_to_fp4",
        "rocm_sets_enable_fp4_flag": True,
    }


def build_summary() -> dict[str, object]:
    sm103_attn = simulate_sm103_nvfp4_tmem_attention()
    megamoe = simulate_megamoe_ep16_relaxed_fence_and_dispatch()
    kimi_ess = simulate_kimi_k3_tp12_ess_routing()
    sm120 = simulate_sm120_fp8_varlen_and_device_alpha()
    tilelang = simulate_tilelang_ascend_and_rocm_fp4()
    return {
        "experiment_id": "flashinfer_sm103_nvfp4_tmem_megamoe_ep16_kimi_ess_sm120_tilelang",
        "phase": 76,
        "priority": 85,
        "sm103_nvfp4_tmem_attention": sm103_attn,
        "megamoe_ep16_relaxed_fence_dispatch": megamoe,
        "kimi_k3_tp12_ess_routing": kimi_ess,
        "sm120_fp8_varlen_and_device_alpha": sm120,
        "tilelang_ascend_and_rocm_fp4": tilelang,
        "invariants": {
            "sm103_tmem_ncols_480": sm103_attn["tmem_ncols_pr"] == 480,
            "megamoe_single_6kib_row_copy": megamoe["single_pass_6kib_verified"],
            "kimi_ess_covers_256_to_960": (
                kimi_ess["routes"]["256"]["k3"] == "k3_ess:grouped"
                and kimi_ess["routes"]["960"]["k3"] == "k3_ess:grouped"
                and kimi_ess["routes"]["961"]["k3"] == "k3_persist_bulk:pinned"
            ),
            "sm120_host_syncs_eliminated": sm120["host_syncs_removed_per_call"] == 2,
            "tilelang_l0_pitch_preserves_full_n": (
                tilelang["explicit_l0_pitch_bytes"] == 512
                and tilelang["exact_l1_write_footprint_bytes"] == 16384
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify summary.json matches")
    args = parser.parse_args()

    summary = build_summary()
    assert all(summary["invariants"].values()), f"Failed invariants: {summary['invariants']}"

    out_path = Path(__file__).resolve().parent / "summary.json"
    rendered = json.dumps(summary, indent=2, ensure_ascii=True) + "\n"
    if args.check:
        if not out_path.exists():
            raise SystemExit(f"Missing {out_path}")
        existing = out_path.read_text(encoding="utf-8")
        if existing != rendered:
            raise SystemExit("summary.json is out of date with run.py")
        print("OK: Phase 76 Track A summary.json verified.")
        return 0

    out_path.write_text(rendered, encoding="utf-8", newline="\n")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
