from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def simulate_bf16_bmm_swap_ab_routing() -> dict[str, Any]:
    """Model FlashInfer #7351: tcgen05 swap-AB programs for decode-row bmm_bf16 on SM100/SM103."""
    routed_shapes = [
        {"batch": 1, "m": 8, "n": 40, "k": 1024, "out": "f32", "b200_prev_us": 3.38, "b200_new_us": 2.94, "gb300_prev_us": 3.23, "gb300_new_us": 2.74},
        {"batch": 2, "m": 33, "n": 1032, "k": 1024, "out": "bf16", "b200_prev_us": 5.50, "b200_new_us": 4.59, "gb300_prev_us": 5.15, "gb300_new_us": 4.22},
        {"batch": 1, "m": 16, "n": 64, "k": 1024, "out": "f16", "b200_prev_us": 3.42, "b200_new_us": 2.88, "gb300_prev_us": 3.30, "gb300_new_us": 2.77},
        {"batch": 2, "m": 8, "n": 1024, "k": 1024, "out": "f16", "b200_prev_us": 4.29, "b200_new_us": 3.95, "gb300_prev_us": 3.94, "gb300_new_us": 3.60},
        {"batch": 2, "m": 8, "n": 1024, "k": 1024, "out": "f32", "b200_prev_us": 4.19, "b200_new_us": 3.90, "gb300_prev_us": 4.03, "gb300_new_us": 3.73},
        {"batch": 4, "m": 16, "n": 1024, "k": 1024, "out": "f16", "b200_prev_us": 5.06, "b200_new_us": 4.88, "gb300_prev_us": 4.82, "gb300_new_us": 4.64},
        {"batch": 4, "m": 16, "n": 1024, "k": 1024, "out": "f32", "b200_prev_us": 5.06, "b200_new_us": 4.90, "gb300_prev_us": 4.85, "gb300_new_us": 4.70},
        {"batch": 2, "m": 8, "n": 1024, "k": 1024, "out": "bf16", "b200_prev_us": 3.92, "b200_new_us": 3.89, "gb300_prev_us": 3.81, "gb300_new_us": 3.71},
        {"batch": 4, "m": 16, "n": 1024, "k": 1024, "out": "bf16", "b200_prev_us": 5.10, "b200_new_us": 4.90, "gb300_prev_us": 4.86, "gb300_new_us": 4.70},
    ]
    rows = []
    for s in routed_shapes:
        b200_speedup = round(s["b200_prev_us"] / s["b200_new_us"], 2)
        gb300_speedup = round(s["gb300_prev_us"] / s["gb300_new_us"], 2)
        rows.append({
            **s,
            "b200_speedup_x": b200_speedup,
            "gb300_speedup_x": gb300_speedup,
            "mma_swapped_m": s["n"],
            "mma_swapped_n": s["m"],
        })
    return {
        "routed_decode_shapes_per_arch": len(rows),
        "architectures": ["sm_100a", "sm_103a"],
        "b200_speedup_range_x": [min(r["b200_speedup_x"] for r in rows), max(r["b200_speedup_x"] for r in rows)],
        "gb300_speedup_range_x": [min(r["gb300_speedup_x"] for r in rows), max(r["gb300_speedup_x"] for r in rows)],
        "shapes": rows,
    }


def simulate_cake_nvfp4_attention() -> dict[str, Any]:
    """Model FlashInfer #7362: Cake NVFP4 attention on SM100, SM103, and SM120."""
    shapes = [
        {"id": "R1", "batch": 1, "seq_len": 4800, "heads": 40, "b200_cake_ms": 0.4940, "b200_fa4_ms": 0.3534, "gb300_cake_ms": 0.3923, "gb300_fa4_ms": 0.2743, "sm120_cake_ms": 1.0082, "sm120_sdpa_ms": 1.4242},
        {"id": "R2", "batch": 1, "seq_len": 32760, "heads": 40, "b200_cake_ms": 16.0399, "b200_fa4_ms": 16.4005, "gb300_cake_ms": 11.1800, "gb300_fa4_ms": 11.8955, "sm120_cake_ms": 33.2754, "sm120_sdpa_ms": 59.6590},
        {"id": "R3", "batch": 1, "seq_len": 75600, "heads": 40, "b200_cake_ms": 84.5186, "b200_fa4_ms": 87.0277, "gb300_cake_ms": 58.4486, "gb300_fa4_ms": 62.9903, "sm120_cake_ms": 172.0325, "sm120_sdpa_ms": 315.4792},
        {"id": "R4", "batch": 1, "seq_len": 32760, "heads": 5, "b200_cake_ms": 1.9515, "b200_fa4_ms": 2.0763, "gb300_cake_ms": 1.4460, "gb300_fa4_ms": 1.4919, "sm120_cake_ms": 4.1421, "sm120_sdpa_ms": 7.5130},
        {"id": "R5", "batch": 1, "seq_len": 75600, "heads": 5, "b200_cake_ms": 10.4680, "b200_fa4_ms": 10.8258, "gb300_cake_ms": 7.3968, "gb300_fa4_ms": 7.9395, "sm120_cake_ms": 21.5641, "sm120_sdpa_ms": 39.7563},
        {"id": "R6", "batch": 2, "seq_len": 32760, "heads": 40, "b200_cake_ms": 32.4710, "b200_fa4_ms": 32.8883, "gb300_cake_ms": 22.3310, "gb300_fa4_ms": 23.9050, "sm120_cake_ms": 66.7597, "sm120_sdpa_ms": 119.1982},
        {"id": "R7", "batch": 1, "seq_len": 32760, "heads": 12, "b200_cake_ms": 4.8390, "b200_fa4_ms": 5.0445, "gb300_cake_ms": 3.3869, "gb300_fa4_ms": 3.5245, "sm120_cake_ms": 10.1126, "sm120_sdpa_ms": 18.1876},
    ]
    evaluated = []
    for s in shapes:
        evaluated.append({
            **s,
            "b200_speedup_x": round(s["b200_fa4_ms"] / s["b200_cake_ms"], 2),
            "gb300_speedup_x": round(s["gb300_fa4_ms"] / s["gb300_cake_ms"], 2),
            "sm120_speedup_x": round(s["sm120_sdpa_ms"] / s["sm120_cake_ms"], 2),
        })
    long_seq = [e for e in evaluated if e["id"] != "R1"]
    return {
        "head_dim": 128,
        "sm100_sm103_chain_kernels": 4,
        "sm120_chain_kernels": 3,
        "b200_r2_r7_speedup_range_x": [min(e["b200_speedup_x"] for e in long_seq), max(e["b200_speedup_x"] for e in long_seq)],
        "gb300_r2_r7_speedup_range_x": [min(e["gb300_speedup_x"] for e in long_seq), max(e["gb300_speedup_x"] for e in long_seq)],
        "sm120_r1_r7_speedup_range_x": [min(e["sm120_speedup_x"] for e in evaluated), max(e["sm120_speedup_x"] for e in evaluated)],
        "r1_quant_overhead_ratio_b200": evaluated[0]["b200_speedup_x"],
        "r1_quant_overhead_ratio_gb300": evaluated[0]["gb300_speedup_x"],
        "shapes": evaluated,
    }


def simulate_hopper_msa_deslop_and_cudnn_guards() -> dict[str, Any]:
    """Model FlashInfer #7332 (Hopper MSA deslop), #7331/#7317 (cuDNN warm stream/version cache), #7320 (SM120 32-bit guard), #7354 (sm_107a floor), and TileLang #3643."""
    # #7320: 32-bit signed offset guard for MiniMax-H3 SM120 varlen attention
    max_elements = (1 << 31) - 1
    heads = 56
    head_dim = 128
    block_n = 128
    row_elements = heads * head_dim
    max_safe_padded_tokens = (max_elements // row_elements // block_n) * block_n
    overflow_tokens = max_safe_padded_tokens + 1
    overflow_padded = ((overflow_tokens + block_n - 1) // block_n) * block_n

    # #7332: Hopper MSA translation-unit folding
    hopper_msa = {
        "programs_registered": 89,
        "base_files": 179,
        "head_files": 68,
        "base_bindings": 89,
        "shared_role_bindings": 15,
        "deleted_bindings": 74,
        "folded_clone_kernels_deleted": 38,
        "shared_kernel_sources": 20,
        "remaining_kernel_units": 51,
        "distinct_arg_plans": 7,
        "lines_added": 2313,
        "lines_removed": 79655,
        "net_lines_delta": 2313 - 79655,
    }

    # #7331 & #7317: cuDNN host overhead
    cudnn_host = {
        "version_predicate_before_us": 0.897,
        "version_predicate_after_us": 0.072,
        "version_predicate_speedup_x": round(0.897 / 0.072, 2),
        "ragged_prefill_enqueue_before_us": 32.796,
        "ragged_prefill_enqueue_after_us": 31.154,
        "ragged_prefill_saved_us": round(32.796 - 31.154, 3),
        "hnd_decode_enqueue_before_us": 35.717,
        "hnd_decode_enqueue_after_us": 32.706,
        "hnd_decode_saved_us": round(35.717 - 32.706, 3),
    }

    # TileLang #3643: named barrier allocator across "shared" and "shared.dyn" scopes
    used_barriers: set[int] = set()
    first_reserved = 1
    scope_allocations: dict[str, int] = {}
    for scope in ("shared", "shared.dyn"):
        bid = first_reserved
        while bid in used_barriers:
            bid += 1
        used_barriers.add(bid)
        scope_allocations[scope] = bid

    return {
        "hopper_msa": hopper_msa,
        "cudnn_host": cudnn_host,
        "minimax_h3_sm120_32bit_guard": {
            "max_int32_elements": max_elements,
            "heads": heads,
            "head_dim": head_dim,
            "max_safe_padded_tokens": max_safe_padded_tokens,
            "overflow_tokens": overflow_tokens,
            "overflow_elements": overflow_padded * row_elements,
            "overflow_detected": overflow_padded * row_elements > max_elements,
        },
        "sm107a_min_cuda_floor": {
            "previous_min_cuda": "13.0",
            "updated_min_cuda": "13.4",
            "updated_manifest_entries": 24,
        },
        "tilelang_barrier_isolation": {
            "shared_barrier_id": scope_allocations["shared"],
            "shared_dyn_barrier_id": scope_allocations["shared.dyn"],
            "disjoint_ids": scope_allocations["shared"] != scope_allocations["shared.dyn"],
        },
    }


def build_summary() -> dict[str, Any]:
    return {
        "experiment_id": "flashinfer_bmm_nvfp4_hopper_msa_phase87",
        "phase": 87,
        "priority": 96,
        "bf16_bmm_swap_ab": simulate_bf16_bmm_swap_ab_routing(),
        "cake_nvfp4_attention": simulate_cake_nvfp4_attention(),
        "hopper_msa_and_host_guards": simulate_hopper_msa_deslop_and_cudnn_guards(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    summary = build_summary()
    out_path = Path(__file__).resolve().parent / "summary.json"
    if args.check:
        assert summary["bf16_bmm_swap_ab"]["routed_decode_shapes_per_arch"] == 9
        assert summary["bf16_bmm_swap_ab"]["gb300_speedup_range_x"] == [1.03, 1.22]
        assert summary["cake_nvfp4_attention"]["gb300_r2_r7_speedup_range_x"] == [1.03, 1.08]
        assert summary["cake_nvfp4_attention"]["sm120_r1_r7_speedup_range_x"] == [1.41, 1.84]
        assert summary["hopper_msa_and_host_guards"]["hopper_msa"]["head_files"] == 68
        assert summary["hopper_msa_and_host_guards"]["minimax_h3_sm120_32bit_guard"]["max_safe_padded_tokens"] == 299520
        assert summary["hopper_msa_and_host_guards"]["tilelang_barrier_isolation"]["disjoint_ids"] is True
        print("OK: flashinfer_bmm_nvfp4_hopper_msa_phase87 verified")
        return 0

    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
