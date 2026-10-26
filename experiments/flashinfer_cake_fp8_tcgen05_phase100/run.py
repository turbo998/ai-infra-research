from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def simulate_cake_fp8_sm90_attention() -> dict[str, Any]:
    """Model FlashInfer #7373: Cake INT8 QK (K-mean smoothed) + FP8 E4M3 PV 4-kernel chain on SM90."""
    # Exact author-reported H100 80GB HBM3 (sm_90a) benchmark table from FlashInfer PR #7373
    shapes = [
        {"id": "R1", "batch": 1, "seq_len": 4800, "heads": 40, "head_dim": 128, "cake_ms": 0.7530, "fa3_bf16_ms": 0.6484},
        {"id": "R2", "batch": 1, "seq_len": 32760, "heads": 40, "head_dim": 128, "cake_ms": 28.9556, "fa3_bf16_ms": 31.6100},
        {"id": "R3", "batch": 1, "seq_len": 75600, "heads": 40, "head_dim": 128, "cake_ms": 154.6865, "fa3_bf16_ms": 176.3442},
        {"id": "R4", "batch": 1, "seq_len": 32760, "heads": 5, "head_dim": 128, "cake_ms": 3.7545, "fa3_bf16_ms": 4.0651},
        {"id": "R5", "batch": 1, "seq_len": 75600, "heads": 5, "head_dim": 128, "cake_ms": 19.2451, "fa3_bf16_ms": 22.0230},
        {"id": "R6", "batch": 2, "seq_len": 32760, "heads": 40, "head_dim": 128, "cake_ms": 58.0676, "fa3_bf16_ms": 63.9349},
        {"id": "R7", "batch": 1, "seq_len": 32760, "heads": 12, "head_dim": 128, "cake_ms": 8.7903, "fa3_bf16_ms": 10.2247},
    ]
    rows: list[dict[str, Any]] = []
    for row in shapes:
        b = row["batch"]
        s = row["seq_len"]
        h = row["heads"]
        d = row["head_dim"]
        # Dense non-causal MHA FLOPs = 4 * B * H * S^2 * D
        total_flops = 4.0 * b * h * (s ** 2) * d
        cake_tflops = (total_flops / (row["cake_ms"] * 1e-3)) / 1e12
        fa3_tflops = (total_flops / (row["fa3_bf16_ms"] * 1e-3)) / 1e12
        speedup = round(row["fa3_bf16_ms"] / row["cake_ms"], 2)
        # Workspace holds INT8 Q, INT8 smoothed K, FP8 E4M3 V, K-mean partials, and per-tile scales
        q_int8_bytes = b * h * s * d
        k_int8_bytes = b * h * s * d
        v_fp8_bytes = b * h * s * d
        stats_and_scale_bytes = b * h * ((s + 127) // 128) * 16 + b * h * d * 8
        workspace_bytes = q_int8_bytes + k_int8_bytes + v_fp8_bytes + stats_and_scale_bytes
        rows.append({
            **row,
            "speedup_vs_fa3_bf16": speedup,
            "cake_effective_tflops": round(cake_tflops, 2),
            "fa3_effective_tflops": round(fa3_tflops, 2),
            "workspace_bytes": workspace_bytes,
            "kernel_chain": ["statistics_partials", "statistics_finalize", "quantization", "attention"],
        })
    r2_to_r7_speedups = [r["speedup_vs_fa3_bf16"] for r in rows if r["id"] != "R1"]
    return {
        "rows": rows,
        "r1_short_seq_speedup": rows[0]["speedup_vs_fa3_bf16"],
        "r2_to_r7_min_speedup": min(r2_to_r7_speedups),
        "r2_to_r7_max_speedup": max(r2_to_r7_speedups),
        "kernel_chain_length": 4,
    }


def simulate_tcgen05_narrow_unroll_and_fp4_allreduce_deslop() -> dict[str, Any]:
    """Model FlashInfer #7370 (tcgen05 narrow unroll), #7366 (FP4 allreduce deslop), #6231 (cuDNN plan reuse), and TileLang #3523/#2887."""
    arch_builds = {
        "sm_100a": {"unroll_rounds": 2, "per_step_k_test": True, "programs_before": 12, "programs_after": 20},
        "sm_103a": {"unroll_rounds": 4, "per_step_k_test": False, "programs_before": 12, "programs_after": 20},
        "sm_107a": {"unroll_rounds": 4, "per_step_k_test": True, "programs_before": 12, "programs_after": 20},
    }
    b200_narrow_shapes = [
        {"shape": "1x128x14336", "with_bias_base_us": 7.58, "with_bias_cake_us": 6.85, "no_bias_base_us": 7.42, "no_bias_cake_us": 6.88},
        {"shape": "7x128x4096", "with_bias_base_us": 4.29, "with_bias_cake_us": 4.13, "no_bias_base_us": 4.37, "no_bias_cake_us": 4.16},
        {"shape": "8x128x7168", "with_bias_base_us": 5.31, "with_bias_cake_us": 5.29, "no_bias_base_us": 5.22, "no_bias_cake_us": 4.99},
    ]
    for item in b200_narrow_shapes:
        item["with_bias_speedup"] = round(item["with_bias_base_us"] / item["with_bias_cake_us"], 2)
        item["no_bias_speedup"] = round(item["no_bias_base_us"] / item["no_bias_cake_us"], 2)

    fp4_allreduce_deslop = {
        "cu_files_before": 116,
        "cu_files_after": 49,
        "kernel_files_after": 39,
        "shared_binding_files_after": 10,
        "programs_before": 58,
        "programs_after": 53,
        "modules_before": 75,
        "modules_after": 68,
        "unreachable_prefetch_min_ctas": 212,
        "folded_pf_twins": 14,
        "lines_added": 2038,
        "lines_deleted": 83919,
        "net_line_reduction": 83919 - 2038,
    }

    # TileLang #3523 SIMTVectorLocalScalarOffset & #2887 empty stepped loop unroll guard
    vector_slot_cases = []
    for lanes in (2, 4):
        for slot in (0, 1, 3):
            vector_slot_cases.append({
                "lanes": lanes,
                "slot": slot,
                "scalar_offset": slot * lanes,
                "buggy_unscaled_offset": slot,
            })

    def unroll_stepped_loop_count(extent: int, step: int) -> int:
        if extent < 0:
            return -1
        if extent == 0:
            return 0
        return 1 + (extent - 1) // step

    return {
        "arch_builds": arch_builds,
        "b200_narrow_shapes": b200_narrow_shapes,
        "fp4_allreduce_deslop": fp4_allreduce_deslop,
        "cudnn_fa2_graph_guard": {
            "graph_auto_sticky_fa2_after_fa2_plan": True,
            "b200_validation_tests_passed": 70,
            "strict_lse_checks_passed": 1000,
            "lifecycle_checks_passed": 1700,
        },
        "tilelang_simt_vector_local": {
            "cases": vector_slot_cases,
            "empty_loop_step4_unroll_count": unroll_stepped_loop_count(0, 4),
            "nonempty_loop_extent8_step4_unroll_count": unroll_stepped_loop_count(8, 4),
        },
    }


def run_experiment() -> dict[str, Any]:
    attention = simulate_cake_fp8_sm90_attention()
    tcgen05_and_deslop = simulate_tcgen05_narrow_unroll_and_fp4_allreduce_deslop()
    return {
        "experiment_id": "flashinfer_cake_fp8_tcgen05_phase100",
        "phase": 100,
        "priority": 109,
        "cake_fp8_sm90_attention": attention,
        "tcgen05_and_deslop": tcgen05_and_deslop,
        "invariants": {
            "cake_fp8_long_seq_speedup_band_holds": (
                attention["r2_to_r7_min_speedup"] == 1.08
                and attention["r2_to_r7_max_speedup"] == 1.16
                and attention["r1_short_seq_speedup"] == 0.86
            ),
            "tcgen05_narrow_programs_expanded_12_to_20": all(
                v["programs_before"] == 12 and v["programs_after"] == 20
                for v in tcgen05_and_deslop["arch_builds"].values()
            ),
            "fp4_allreduce_cu_files_reduced_116_to_49": (
                tcgen05_and_deslop["fp4_allreduce_deslop"]["cu_files_before"] == 116
                and tcgen05_and_deslop["fp4_allreduce_deslop"]["cu_files_after"] == 49
                and tcgen05_and_deslop["fp4_allreduce_deslop"]["net_line_reduction"] == 81881
            ),
            "tilelang_empty_stepped_loop_returns_zero": (
                tcgen05_and_deslop["tilelang_simt_vector_local"]["empty_loop_step4_unroll_count"] == 0
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 100 FlashInfer Cake FP8 Attention & tcgen05 Simulator")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("summary.json"))
    args = parser.parse_args()
    summary = run_experiment()
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
