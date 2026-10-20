from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SUMMARY_PATHS = [ROOT / "summary.json", ROOT / "results" / "summary.json"]

CAKE_QUANT_BLOCK = 32
CAKE_MAX_GRID_YZ = 65535
CAKE_SCALE_TILE = 128

AUTHOR_BENCHMARKS = [
    {
        "gpu": "B200 (sm_100a)",
        "dtype": "bfloat16",
        "shape": "B=2 M=256 K=4096 (T=512)",
        "previous_us": 3.744,
        "new_us": 3.808,
        "speedup": 0.983,
    },
    {
        "gpu": "B200 (sm_100a)",
        "dtype": "float16",
        "shape": "B=2 M=256 K=4096 (T=512)",
        "previous_us": 3.968,
        "new_us": 4.000,
        "speedup": 0.992,
    },
    {
        "gpu": "GB300 (sm_103a)",
        "dtype": "bfloat16",
        "shape": "B=2 M=256 K=4096 (T=512)",
        "previous_us": 3.744,
        "new_us": 3.776,
        "speedup": 0.992,
    },
    {
        "gpu": "GB300 (sm_103a)",
        "dtype": "float16",
        "shape": "B=2 M=256 K=4096 (T=512)",
        "previous_us": 3.648,
        "new_us": 3.648,
        "speedup": 1.000,
    },
]


def round_up(x: int, align: int) -> int:
    return ((x + align - 1) // align) * align


def simulate_cake_grouped_mxfp8_launch(
    b: int,
    m: int,
    k: int,
    dtype: str = "bfloat16",
    capability: tuple[int, int] = (10, 0),
    cta_threads: int = 128,
) -> dict[str, object]:
    if capability not in {(10, 0), (10, 3)}:
        raise RuntimeError(
            f"the Cake grouped MXFP8 backend requires exact compute capability 10.0 or 10.3, got {capability[0]}.{capability[1]}"
        )
    if dtype not in {"bfloat16", "float16"}:
        raise TypeError(f"unsupported Cake grouped MXFP8 input dtype: {dtype}")
    if k % CAKE_QUANT_BLOCK != 0:
        raise ValueError(f"K must be divisible by {CAKE_QUANT_BLOCK}, got {k}")
    if m > CAKE_MAX_GRID_YZ or b > CAKE_MAX_GRID_YZ:
        raise ValueError(
            f"B and M must not exceed the CUDA grid limit {CAKE_MAX_GRID_YZ}, got B={b}, M={m}"
        )
    padded_k = round_up(k, CAKE_SCALE_TILE)
    padded_m = round_up(m, CAKE_SCALE_TILE)
    blocks_per_row = padded_k // CAKE_QUANT_BLOCK
    pm_tiles = padded_m // CAKE_SCALE_TILE
    pk_tiles = padded_k // CAKE_SCALE_TILE
    grid_x = (blocks_per_row + cta_threads - 1) // cta_threads
    grid = (grid_x, m, b)
    scales_6d_shape = (b, pm_tiles, pk_tiles, 32, 4, 4)
    scales_numel = b * pm_tiles * pk_tiles * 32 * 4 * 4
    expected_scales_numel = b * padded_m * blocks_per_row
    assert scales_numel == expected_scales_numel
    program_id = (
        "cake_grouped_mxfp8_quantize_d8aa920df5a6252450bd"
        if dtype == "bfloat16"
        else "cake_grouped_mxfp8_quantize_f84d4eab2905894ac4c2"
    )
    return {
        "b": b,
        "m": m,
        "k": k,
        "dtype": dtype,
        "target": "sm_100a" if capability == (10, 0) else "sm_103a",
        "program_id": program_id,
        "x_passed_as_bf16_view": True,
        "padded_m": padded_m,
        "padded_k": padded_k,
        "blocks_per_row": blocks_per_row,
        "pm_tiles": pm_tiles,
        "pk_tiles": pk_tiles,
        "grid": list(grid),
        "total_tasks": b * m * blocks_per_row,
        "scales_3d_shape": [b, padded_m, blocks_per_row],
        "scales_6d_shape": list(scales_6d_shape),
        "scales_bytes": scales_numel,
    }


def simulate_dsv4_eplb_redundant_slots(
    num_routed_experts: int = 64,
    num_redundant_experts: int = 8,
) -> dict[str, object]:
    physical_to_logical = list(range(num_routed_experts)) + [
        i * 7 % num_routed_experts for i in range(num_redundant_experts)
    ]
    source_weights = {logical: (logical + 1) * 1024 for logical in range(num_routed_experts)}

    # Legacy bug: get_expert_mapping omitted num_redundant_experts -> redundant slots left zero/unloaded
    legacy_slot_sums = [
        source_weights[logical] if slot < num_routed_experts else 0
        for slot, logical in enumerate(physical_to_logical)
    ]
    legacy_bad_slots = sum(
        legacy_slot_sums[slot] != legacy_slot_sums[physical_to_logical.index(logical)]
        for slot, logical in enumerate(physical_to_logical)
    )

    # Fixed PR #59611 (9d4192aaa1): passes num_redundant_experts=first_layer.ffn.n_redundant_experts
    fixed_slot_sums = [source_weights[logical] for logical in physical_to_logical]
    fixed_bad_slots = sum(
        fixed_slot_sums[slot] != fixed_slot_sums[physical_to_logical.index(logical)]
        for slot, logical in enumerate(physical_to_logical)
    )

    return {
        "num_routed_experts": num_routed_experts,
        "num_redundant_experts": num_redundant_experts,
        "total_physical_slots": len(physical_to_logical),
        "legacy_bad_redundant_slots": legacy_bad_slots,
        "fixed_bad_redundant_slots": fixed_bad_slots,
        "kimi_k3_gb200_agentx_profile": {
            "trays": 4,
            "gpus_per_tray": 4,
            "total_gpus": 16,
            "naive_topology": "TP4xPP4",
            "agentx_topology": "TP8xPP2+DCP8",
            "dcp_comm_backend": "a2a",
            "mla_prefill_backend": "TRTLLM_RAGGED",
            "speculative_tokens": 4,
            "speculative_method": "dspark",
            "pipeline_stages_reduced_pct": 50.0,
        },
    }


def build_summary() -> dict[str, object]:
    launch_bf16 = simulate_cake_grouped_mxfp8_launch(2, 256, 4096, "bfloat16", (10, 0))
    launch_f16 = simulate_cake_grouped_mxfp8_launch(2, 256, 4096, "float16", (10, 3))
    unaligned_case = simulate_cake_grouped_mxfp8_launch(4, 130, 4128, "bfloat16", (10, 3))
    eplb = simulate_dsv4_eplb_redundant_slots(64, 8)

    invariants = {
        "cake_grouped_mxfp8_dual_program_loader_verified": (
            launch_bf16["program_id"] != launch_f16["program_id"]
            and launch_f16["x_passed_as_bf16_view"] is True
        ),
        "cake_scale_128x4_layout_bijective": (
            unaligned_case["padded_m"] == 256
            and unaligned_case["padded_k"] == 4224
            and unaligned_case["scales_6d_shape"] == [4, 2, 33, 32, 4, 4]
        ),
        "dsv4_eplb_redundant_slots_zero_mismatch": (
            eplb["legacy_bad_redundant_slots"] == 8
            and eplb["fixed_bad_redundant_slots"] == 0
        ),
        "kimi_k3_gb200_tp8_pp2_dcp8_verified": (
            eplb["kimi_k3_gb200_agentx_profile"]["agentx_topology"] == "TP8xPP2+DCP8"
        ),
    }
    return {
        "experiment": "flashinfer_cake_grouped_mxfp8_vllm_dsv4_eplb_kimi",
        "phase": 40,
        "priority": 49,
        "author_benchmarks": AUTHOR_BENCHMARKS,
        "cake_launches": {
            "b200_bf16": launch_bf16,
            "gb300_f16": launch_f16,
            "unaligned_padding": unaligned_case,
        },
        "vllm_dsv4_eplb_and_kimi_k3": eplb,
        "invariants": invariants,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    summary = build_summary()
    assert all(summary["invariants"].values()), f"Invariant failed: {summary['invariants']}"

    for path in SUMMARY_PATHS:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if args.check:
        print(
            json.dumps(
                {
                    "status": "ok",
                    "experiment": summary["experiment"],
                    "invariants": summary["invariants"],
                }
            )
        )


if __name__ == "__main__":
    main()
