"""Phase 101 Experiment 1: FlashInfer Cake MiniMax-H3 N-Split Tail, Kimi-K3 Fused Router Arm M Co-Resident CTA Guard & SM120 Sage Attention.

Models:
1. MiniMax-H3 Attention Out-Projection N-Split Tail Schedule (`bf82f4d797`, PR #7363):
   - BF16 (`minimax_h3_out_proj`) and MXFP8 (`minimax_h3_out_proj_mxfp8`, E4M3 data + UE8M0 block-32 scales) on SM100/SM103.
   - Eligibility rules (`rows >= 16`, waves <= 10 on SM103 / <= 9 on SM100 for BF16; `rows >= 129`, waves <= 26 on SM103 / <= 17 on SM100 for MXFP8, with a partial last wave fitting in <= half a wave) and half-width 256x128 tail tile scheduling.
   - Reproduces reported GB300/GB200 latency tables and computes wave-efficiency recovery.
2. Kimi-K3 Fused Router Arm M Co-Resident CTA Guard (`e18eb1d2ee`, PR #7365):
   - 896 experts, top-16, `OWNER_CTAS = 128` plan-owner CTAs (7 experts per CTA: `7b .. 7b+6` for Arm M vs strided owners for Arms L/LP/GW).
   - Persistent capacity boundary check: CC 10.0 (`3 CTAs/SM` -> 42 SMs rejected / 43 SMs admitted), CC 10.3 (`4 CTAs/SM` -> 31 SMs rejected / 32 SMs admitted).
3. SM120 Block-Sparse Sage Attention First-Half P Pre-Computation (`967a8ed858`, PR #7364) & TileLang Cast/Target Validation (`e84ce3de8a`, `f1a34efea6`):
   - 20-program metadata specialization (`2^5` pruned to 20 valid programs), 8-shape cold-L2 latency and speedup analysis, and TileLang `cvt.rs` tuple validation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


MINIMAX_H3_HIDDEN = 5376  # 21 column tiles of 256
TILE_M_PAIR = 256  # pair of 128-row tiles
TILE_N_FULL = 256
TILE_N_HALF = 128


def m_tile_pairs(rows: int) -> int:
    """Number of 256-row tile pairs covering `rows`."""
    m_tiles_128 = math.ceil(rows / 128)
    if m_tiles_128 % 2 == 1:
        m_tiles_128 += 1
    return m_tiles_128 // 2


def evaluate_minimax_h3_tail_split() -> dict[str, Any]:
    """Model MiniMax-H3 BF16 and MXFP8 N-split tail schedule on SM100 and SM103."""
    n_tiles = MINIMAX_H3_HIDDEN // TILE_N_FULL  # 21
    co_resident_pairs = 76  # 152 CTAs = 76 2-CTA pairs on SM100/SM103 persistent schedule

    # Reported GB300 (sm_103a) and GB200 (sm_100a) tail-split benchmark points from PR #7363
    bf16_gb300_cases = [
        {"P": 4, "M": 8368, "before_us": 383.3, "after_us": 378.6, "speedup": 1.012},
        {"P": 8, "M": 4816, "before_us": 227.7, "after_us": 218.9, "speedup": 1.040},
        {"P": 8, "M": 4823, "before_us": 228.0, "after_us": 219.0, "speedup": 1.041},
        {"P": 8, "M": 4824, "before_us": 228.5, "after_us": 220.4, "speedup": 1.037},
        {"P": 8, "M": 4825, "before_us": 227.6, "after_us": 219.0, "speedup": 1.039},
        {"P": 8, "M": 4832, "before_us": 227.4, "after_us": 218.6, "speedup": 1.041},
        {"P": 8, "M": 7368, "before_us": 342.4, "after_us": 332.4, "speedup": 1.030},
    ]

    mxfp8_gb300_cases = [
        {"P": 2, "M": 16736, "before_us": 467.0, "after_us": 458.7, "speedup": 1.018},
        {"P": 4, "M": 8368, "before_us": 246.8, "after_us": 238.8, "speedup": 1.033},
        {"P": 4, "M": 9632, "before_us": 270.5, "after_us": 264.9, "speedup": 1.021},
        {"P": 4, "M": 9647, "before_us": 270.4, "after_us": 264.8, "speedup": 1.021},
        {"P": 4, "M": 9648, "before_us": 271.1, "after_us": 265.8, "speedup": 1.020},
        {"P": 4, "M": 9649, "before_us": 269.5, "after_us": 264.5, "speedup": 1.019},
        {"P": 4, "M": 9664, "before_us": 270.1, "after_us": 264.7, "speedup": 1.021},
        {"P": 4, "M": 12192, "before_us": 339.2, "after_us": 331.9, "speedup": 1.022},
        {"P": 4, "M": 14736, "before_us": 413.3, "after_us": 404.7, "speedup": 1.021},
        {"P": 4, "M": 18560, "before_us": 512.4, "after_us": 503.2, "speedup": 1.018},
        {"P": 8, "M": 4816, "before_us": 145.0, "after_us": 137.2, "speedup": 1.057},
        {"P": 8, "M": 4823, "before_us": 144.8, "after_us": 137.0, "speedup": 1.057},
        {"P": 8, "M": 4824, "before_us": 145.0, "after_us": 137.6, "speedup": 1.053},
        {"P": 8, "M": 4825, "before_us": 144.7, "after_us": 136.9, "speedup": 1.057},
        {"P": 8, "M": 4832, "before_us": 144.9, "after_us": 137.4, "speedup": 1.055},
        {"P": 8, "M": 7368, "before_us": 217.0, "after_us": 209.6, "speedup": 1.035},
        {"P": 8, "M": 9280, "before_us": 267.5, "after_us": 260.9, "speedup": 1.025},
    ]

    wave_analysis = []
    for rows in (1, 4824, 7368, 8368, 9648, 16736):
        row_pairs = m_tile_pairs(rows)
        total_tile_pairs = row_pairs * n_tiles
        full_waves = total_tile_pairs // co_resident_pairs
        tail_pairs = total_tile_pairs % co_resident_pairs
        total_waves = math.ceil(total_tile_pairs / co_resident_pairs)
        half_wave_eligible = 0 < tail_pairs <= (co_resident_pairs // 2)
        bf16_sm103_eligible = rows >= 16 and half_wave_eligible and total_waves <= 10
        bf16_sm100_eligible = rows >= 16 and half_wave_eligible and total_waves <= 9
        mxfp8_sm103_eligible = rows >= 129 and half_wave_eligible and total_waves <= 26
        mxfp8_sm100_eligible = rows >= 129 and half_wave_eligible and total_waves <= 17
        # Effective wave cost when tail tiles run as half-width 256x128 units (~0.55x tile time)
        whole_tile_wave_units = float(total_waves)
        split_tail_wave_units = full_waves + (0.55 if half_wave_eligible else (1.0 if tail_pairs > 0 else 0.0))
        wave_analysis.append(
            {
                "rows": rows,
                "row_tile_pairs": row_pairs,
                "total_tile_pairs": total_tile_pairs,
                "full_waves": full_waves,
                "tail_tile_pairs": tail_pairs,
                "total_waves": total_waves,
                "half_wave_eligible": half_wave_eligible,
                "bf16_sm103_eligible": bf16_sm103_eligible,
                "bf16_sm100_eligible": bf16_sm100_eligible,
                "mxfp8_sm103_eligible": mxfp8_sm103_eligible,
                "mxfp8_sm100_eligible": mxfp8_sm100_eligible,
                "whole_tile_wave_units": round(whole_tile_wave_units, 4),
                "split_tail_wave_units": round(split_tail_wave_units, 4),
                "theoretical_wave_speedup": round(whole_tile_wave_units / split_tail_wave_units, 4),
            }
        )

    bf16_geomean = math.exp(sum(math.log(c["before_us"] / c["after_us"]) for c in bf16_gb300_cases) / len(bf16_gb300_cases))
    mxfp8_geomean = math.exp(sum(math.log(c["before_us"] / c["after_us"]) for c in mxfp8_gb300_cases) / len(mxfp8_gb300_cases))

    return {
        "hidden_dim": MINIMAX_H3_HIDDEN,
        "n_tiles_256": n_tiles,
        "co_resident_tile_pairs": co_resident_pairs,
        "bf16_gb300_tail_cases_count": len(bf16_gb300_cases),
        "bf16_gb300_geomean_speedup": round(bf16_geomean, 4),
        "bf16_gb300_max_speedup": max(c["speedup"] for c in bf16_gb300_cases),
        "mxfp8_gb300_tail_cases_count": len(mxfp8_gb300_cases),
        "mxfp8_gb300_geomean_speedup": round(mxfp8_geomean, 4),
        "mxfp8_gb300_max_speedup": max(c["speedup"] for c in mxfp8_gb300_cases),
        "wave_analysis": wave_analysis,
        "bf16_gb300_cases": bf16_gb300_cases,
        "mxfp8_gb300_cases": mxfp8_gb300_cases,
    }


def evaluate_kimi_k3_router_arm_m_guard() -> dict[str, Any]:
    """Model Kimi-K3 fused MoE router Arm M co-resident plan-owner CTA boundary (`e18eb1d2ee`, PR #7365)."""
    num_experts = 896
    threads = 224
    num_warps = threads // 32  # 7 warps
    owner_ctas = num_experts // num_warps  # 128 plan-owner CTAs
    ctas_per_sm = {(10, 0): 3, (10, 3): 4}

    device_sweeps = []
    for cc, sm_count in [
        ((10, 0), 32),
        ((10, 0), 42),
        ((10, 0), 43),
        ((10, 0), 148),
        ((10, 3), 24),
        ((10, 3), 31),
        ((10, 3), 32),
        ((10, 3), 160),
    ]:
        persistent_cap = ctas_per_sm[cc] * sm_count
        admits_arm_m = persistent_cap >= owner_ctas
        # Without PR #7365 guard, Arm M launched min(num_tokens, persistent_cap) CTAs,
        # where CTA b < persistent_cap only published experts 7b..7b+6.
        published_experts_unguarded = min(persistent_cap, owner_ctas) * num_warps
        unwritten_experts_unguarded = num_experts - published_experts_unguarded
        # Strided arms L / LP / GW stride over experts by active owner count, so 100% of experts are published.
        device_sweeps.append(
            {
                "compute_capability": f"sm_{cc[0]}{cc[1]}a",
                "sm_count": sm_count,
                "ctas_per_sm": ctas_per_sm[cc],
                "persistent_grid_cap": persistent_cap,
                "admits_arm_m": admits_arm_m,
                "arm_m_grid_256_tokens": min(256, persistent_cap) if admits_arm_m else None,
                "arm_l_grid_128_tokens": min(128, persistent_cap),
                "unguarded_published_experts": published_experts_unguarded,
                "unguarded_unwritten_experts": unwritten_experts_unguarded,
                "strided_l_gw_published_experts": num_experts,
            }
        )

    return {
        "num_experts": num_experts,
        "top_k": 16,
        "threads_per_cta": threads,
        "num_warps_per_cta": num_warps,
        "owner_ctas_required": owner_ctas,
        "min_sm_cc100": math.ceil(owner_ctas / ctas_per_sm[(10, 0)]),  # 43
        "min_sm_cc103": math.ceil(owner_ctas / ctas_per_sm[(10, 3)]),  # 32
        "device_sweeps": device_sweeps,
    }


def evaluate_sm120_sage_and_tilelang_guards() -> dict[str, Any]:
    """Model SM120 Block-Sparse Sage Attention (`967a8ed858`) and TileLang cast/arch guards (`e84ce3de8a`, `f1a34efea6`)."""
    sage_cases = [
        {"batch": 8, "heads": 32, "seq_len": 1024, "selected_kv_blocks": 15, "total_kv_blocks": 16, "baseline_ms": 0.2141, "opt_ms": 0.2121, "speedup": 1.0100},
        {"batch": 8, "heads": 32, "seq_len": 2048, "selected_kv_blocks": 16, "total_kv_blocks": 32, "baseline_ms": 0.4290, "opt_ms": 0.4256, "speedup": 1.0075},
        {"batch": 8, "heads": 32, "seq_len": 4096, "selected_kv_blocks": 7, "total_kv_blocks": 64, "baseline_ms": 0.4783, "opt_ms": 0.4769, "speedup": 1.0031},
        {"batch": 8, "heads": 32, "seq_len": 8192, "selected_kv_blocks": 116, "total_kv_blocks": 128, "baseline_ms": 10.9677, "opt_ms": 10.9274, "speedup": 1.0037},
        {"batch": 8, "heads": 32, "seq_len": 16384, "selected_kv_blocks": 128, "total_kv_blocks": 256, "baseline_ms": 24.2547, "opt_ms": 24.1733, "speedup": 1.0033},
        {"batch": 8, "heads": 32, "seq_len": 32768, "selected_kv_blocks": 52, "total_kv_blocks": 512, "baseline_ms": 20.0529, "opt_ms": 19.9868, "speedup": 1.0033},
        {"batch": 4, "heads": 32, "seq_len": 65536, "selected_kv_blocks": 512, "total_kv_blocks": 1024, "baseline_ms": 202.6229, "opt_ms": 201.9672, "speedup": 1.0034},
        {"batch": 4, "heads": 32, "seq_len": 65536, "selected_kv_blocks": 103, "total_kv_blocks": 1024, "baseline_ms": 39.2189, "opt_ms": 39.0776, "speedup": 1.0036},
    ]

    for c in sage_cases:
        c["density_ratio"] = round(c["selected_kv_blocks"] / c["total_kv_blocks"], 4)
        c["latency_saved_us"] = round((c["baseline_ms"] - c["opt_ms"]) * 1000.0, 2)

    # TileLang cast (src_dtype, dst_dtype, round_mode, has_rbits) validation (`e84ce3de8a`, PR #3350)
    cast_matrix = [
        {"src": "float32x2", "dst": "bfloat16x2", "round": "rs", "rbits": True, "accepted": True},
        {"src": "float32x2", "dst": "float16x2", "round": "rs", "rbits": True, "accepted": True},
        {"src": "float32x2", "dst": "fp8_e4m3x2", "round": "rs", "rbits": True, "accepted": True},
        {"src": "float32", "dst": "e8m0", "round": "rs", "rbits": True, "accepted": False},
        {"src": "bfloat16x2", "dst": "fp8_e4m3x2", "round": "rs", "rbits": True, "accepted": False},
        {"src": "float32x2", "dst": "bfloat16x2", "round": "rn", "rbits": False, "accepted": False},
    ]

    return {
        "specialized_programs": 20,
        "metadata_flags": 5,
        "sage_cases": sage_cases,
        "max_speedup": max(c["speedup"] for c in sage_cases),
        "min_speedup": min(c["speedup"] for c in sage_cases),
        "tilelang_cast_validation_matrix": cast_matrix,
        "tilelang_accepted_rs_casts": sum(1 for x in cast_matrix if x["accepted"]),
        "tilelang_rejected_invalid_casts": sum(1 for x in cast_matrix if not x["accepted"]),
    }


def run_suite() -> dict[str, Any]:
    minimax = evaluate_minimax_h3_tail_split()
    kimi_router = evaluate_kimi_k3_router_arm_m_guard()
    sage_tilelang = evaluate_sm120_sage_and_tilelang_guards()
    return {
        "experiment": "flashinfer_minimax_h3_tail_kimi_router_phase101",
        "phase": 101,
        "priority": 110,
        "provenance": "own-offline",
        "minimax_h3_tail_split": minimax,
        "kimi_k3_router_arm_m": kimi_router,
        "sm120_sage_and_tilelang": sage_tilelang,
        "headline_metrics": {
            "minimax_bf16_gb300_geomean_speedup": minimax["bf16_gb300_geomean_speedup"],
            "minimax_mxfp8_gb300_geomean_speedup": minimax["mxfp8_gb300_geomean_speedup"],
            "minimax_mxfp8_gb300_peak_speedup": minimax["mxfp8_gb300_max_speedup"],
            "kimi_k3_owner_ctas_required": kimi_router["owner_ctas_required"],
            "kimi_k3_min_sm_cc100": kimi_router["min_sm_cc100"],
            "kimi_k3_min_sm_cc103": kimi_router["min_sm_cc103"],
            "sm120_sage_peak_speedup": sage_tilelang["max_speedup"],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify deterministic invariants")
    args = parser.parse_args()

    summary = run_suite()
    out_path = Path(__file__).resolve().parent / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if args.check:
        assert summary["headline_metrics"]["minimax_bf16_gb300_geomean_speedup"] == 1.0343
        assert summary["headline_metrics"]["minimax_mxfp8_gb300_geomean_speedup"] == 1.0325
        assert summary["headline_metrics"]["minimax_mxfp8_gb300_peak_speedup"] == 1.057
        assert summary["headline_metrics"]["kimi_k3_owner_ctas_required"] == 128
        assert summary["headline_metrics"]["kimi_k3_min_sm_cc100"] == 43
        assert summary["headline_metrics"]["kimi_k3_min_sm_cc103"] == 32
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
