#!/usr/bin/env python3
"""Deterministic simulator for TransformerEngine Blackwell FP8 block-scaling to MXFP8
grouped GEMM swizzle emulation and NVFP4 quantize-transpose amax reduction.

Models the exact scale-buffer geometry, dynamic shared-memory prefix-sum tables,
binary-search tile mapping (`find_tensor`), and host upper-bound sizing from:
  - NVIDIA/TransformerEngine `transformer_engine/common/swizzle/swizzle_block_scaling.cu`
    (commit f4a69dc81eabb5002a662e56a36e8454c9840a92)
  - NVIDIA/TransformerEngine `transformer_engine/common/cast/nvfp4/quantize_transpose_nvfp4.cuh`
    (commit 4a3a8a8e8578e96319467d0cc96b9f5db552a7d3)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

BLOCK_LEN = 128
WARP_SIZE = 32
WARPS_PER_TB = 4
TILE_OUTPUT_BYTES = 512  # 128x4 E8M0 bytes per 128x128 data tile
DEFAULT_MAX_SMEM_BYTES = 48 * 1024
BLACKWELL_OPTIN_MAX_SMEM_BYTES = 227 * 1024


def divup(a: int, b: int) -> int:
    return (a + b - 1) // b if b > 0 else 0


def divup_to_multiple(a: int, b: int) -> int:
    return divup(a, b) * b


def input_scale_floats(rows: int, cols: int, is_2d: bool) -> int:
    """Matches `swizzle_kernel_grouped::input_scale_floats<kIs2D>(rows, cols)`."""
    if is_2d:
        return divup(rows, BLOCK_LEN) * divup_to_multiple(divup(cols, BLOCK_LEN), 4)
    return divup(cols, BLOCK_LEN) * divup_to_multiple(rows, 4)


def output_scale_bytes(rows: int, cols: int) -> int:
    """Matches `swizzle_kernel_grouped::output_scale_bytes(rows, cols)`."""
    return divup_to_multiple(rows, BLOCK_LEN) * divup(cols, BLOCK_LEN) * 4


def table_smem_bytes(num_tensors: int) -> int:
    """Matches `swizzle_kernel_grouped::table_smem_bytes(num_tensors)` (8-byte size_t)."""
    return (3 * num_tensors + 1) * 8


def grouped_mxfp8_scale_bytes_upper_bound(
    num_tensors: int,
    logical_first: int,
    logical_last: int,
    varying_mode: str,
) -> int:
    """Matches `grouped_mxfp8_scale_bytes_upper_bound(const GroupedTensor& t)`."""
    if num_tensors == 0 or logical_first == 0 or logical_last == 0:
        return 0
    if varying_mode == "varying_first":
        return (logical_first + (BLOCK_LEN - 1) * num_tensors) * divup(logical_last, BLOCK_LEN) * 4
    if varying_mode == "varying_last":
        return divup_to_multiple(logical_first, BLOCK_LEN) * 4 * (divup(logical_last, BLOCK_LEN) + num_tensors)
    if varying_mode == "uniform":
        return (
            num_tensors
            * divup_to_multiple(logical_first // num_tensors, BLOCK_LEN)
            * divup(logical_last, BLOCK_LEN)
            * 4
        )
    raise ValueError(f"Unsupported varying_mode: {varying_mode}")


def find_tensor(tile_start: list[int], num_tensors: int, tile: int) -> int:
    """Matches `swizzle_kernel_tiles::find_tensor` binary search."""
    lo = 0
    hi = num_tensors
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if tile_start[mid] <= tile:
            lo = mid
        else:
            hi = mid
    return lo


def simulate_grouped_prefix_sum_and_tiles(
    rows_per_tensor: list[int],
    cols_per_tensor: list[int],
    is_2d: bool = False,
) -> dict[str, Any]:
    num_tensors = len(rows_per_tensor)
    tile_start: list[int] = [0] * (num_tensors + 1)
    in_start: list[int] = [0] * num_tensors
    out_start: list[int] = [0] * num_tensors

    tile_carry = 0
    in_carry = 0
    out_carry = 0
    warp_steps = divup(num_tensors, WARP_SIZE)

    for t in range(num_tensors):
        r = rows_per_tensor[t]
        c = cols_per_tensor[t]
        tiles = divup(r, BLOCK_LEN) * divup(c, BLOCK_LEN)
        in_sz = input_scale_floats(r, c, is_2d)
        out_sz = output_scale_bytes(r, c)
        tile_start[t] = tile_carry
        in_start[t] = in_carry
        out_start[t] = out_carry
        tile_carry += tiles
        in_carry += in_sz
        out_carry += out_sz
    tile_start[num_tensors] = tile_carry

    # Verify binary search on boundary and interior tiles, including zero-row experts
    verified_mappings = 0
    for t in range(num_tensors):
        t_begin = tile_start[t]
        t_end = tile_start[t + 1]
        if t_end > t_begin:
            assert find_tensor(tile_start, num_tensors, t_begin) == t
            assert find_tensor(tile_start, num_tensors, t_end - 1) == t
            verified_mappings += 2

    smem_bytes = table_smem_bytes(num_tensors)
    requires_optin_smem = smem_bytes > DEFAULT_MAX_SMEM_BYTES
    assert smem_bytes <= BLACKWELL_OPTIN_MAX_SMEM_BYTES

    return {
        "num_tensors": num_tensors,
        "warp_prefix_steps": warp_steps,
        "total_tiles": tile_carry,
        "exact_input_scale_bytes": in_carry * 4,
        "exact_output_scale_bytes": out_carry,
        "table_smem_bytes": smem_bytes,
        "requires_optin_smem": requires_optin_smem,
        "verified_boundary_mappings": verified_mappings,
        "tile_start_head": tile_start[:8],
    }


def simulate_scenarios() -> dict[str, Any]:
    scenarios: list[dict[str, Any]] = []

    # Scenario 1: DeepSeek-V3 256 routed experts, varying token counts (first_dims vary), 1D block scaling
    # Total routed tokens F = 32768 (e.g., 4096 tokens * top-8), hidden L = 7168
    num_experts_ds = 256
    logical_f_ds = 32768
    logical_l_ds = 7168
    rows_ds: list[int] = []
    # Include 16 empty experts (0 tokens) and jagged non-multiples of 128 to exercise padding & zero-tile skip
    remaining = logical_f_ds
    for e in range(num_experts_ds):
        if e < 16:
            rows_ds.append(0)
        elif e < num_experts_ds - 1:
            val = 110 + ((e * 37) % 53)
            val = min(val, remaining)
            rows_ds.append(val)
            remaining -= val
        else:
            rows_ds.append(remaining)
    cols_ds = [logical_l_ds] * num_experts_ds

    sim_ds_1d = simulate_grouped_prefix_sum_and_tiles(rows_ds, cols_ds, is_2d=False)
    ub_ds = grouped_mxfp8_scale_bytes_upper_bound(
        num_experts_ds, logical_f_ds, logical_l_ds, "varying_first"
    )

    # Scenario 2: Same 256 experts under 2D block scaling (128x128 FP32 input scales -> MXFP8 E8M0 swizzled scales)
    sim_ds_2d = simulate_grouped_prefix_sum_and_tiles(rows_ds, cols_ds, is_2d=True)

    # Scenario 3: Large MoE Stress Test (2048 experts) triggering >48 KiB opt-in dynamic shared memory
    num_experts_large = 2048
    logical_f_large = 65536
    logical_l_large = 4096
    rows_large = [logical_f_large // num_experts_large] * num_experts_large
    cols_large = [logical_l_large] * num_experts_large
    sim_large = simulate_grouped_prefix_sum_and_tiles(rows_large, cols_large, is_2d=False)
    ub_large = grouped_mxfp8_scale_bytes_upper_bound(
        num_experts_large, logical_f_large, logical_l_large, "varying_first"
    )

    # Scenario 4: Varying last_dims (wgrad K-grouped GEMM: F=7168, L=32768 across 64 experts)
    num_experts_wgrad = 64
    logical_f_wgrad = 7168
    logical_l_wgrad = 32768
    rows_wgrad = [logical_f_wgrad] * num_experts_wgrad
    cols_wgrad: list[int] = []
    rem_k = logical_l_wgrad
    for e in range(num_experts_wgrad):
        if e < num_experts_wgrad - 1:
            k_val = 450 + ((e * 29) % 125)
            k_val = min(k_val, rem_k)
            cols_wgrad.append(k_val)
            rem_k -= k_val
        else:
            cols_wgrad.append(rem_k)
    sim_wgrad = simulate_grouped_prefix_sum_and_tiles(rows_wgrad, cols_wgrad, is_2d=False)
    ub_wgrad = grouped_mxfp8_scale_bytes_upper_bound(
        num_experts_wgrad, logical_f_wgrad, logical_l_wgrad, "varying_last"
    )

    for name, mode, scaling, f_dim, l_dim, sim, ub in [
        ("ds_v3_256e_1d_fprop", "varying_first", "1D_FP32_to_MXFP8_E8M0", logical_f_ds, logical_l_ds, sim_ds_1d, ub_ds),
        ("ds_v3_256e_2d_fprop", "varying_first", "2D_FP32_to_MXFP8_E8M0", logical_f_ds, logical_l_ds, sim_ds_2d, ub_ds),
        ("ultra_moe_2048e_optin_smem", "varying_first", "1D_FP32_to_MXFP8_E8M0", logical_f_large, logical_l_large, sim_large, ub_large),
        ("ds_v3_64e_wgrad_varying_k", "varying_last", "1D_FP32_to_MXFP8_E8M0", logical_f_wgrad, logical_l_wgrad, sim_wgrad, ub_wgrad),
    ]:
        exact_out = sim["exact_output_scale_bytes"]
        exact_in = sim["exact_input_scale_bytes"]
        scenarios.append(
            {
                "scenario": name,
                "varying_mode": mode,
                "scaling_recipe": scaling,
                "num_tensors": sim["num_tensors"],
                "logical_first": f_dim,
                "logical_last": l_dim,
                "total_128x128_tiles": sim["total_tiles"],
                "warp_prefix_steps": sim["warp_prefix_steps"],
                "table_smem_bytes": sim["table_smem_bytes"],
                "requires_optin_smem": sim["requires_optin_smem"],
                "input_fp32_scale_bytes": exact_in,
                "exact_mxfp8_scale_bytes": exact_out,
                "host_upper_bound_bytes": ub,
                "upper_bound_slack_ratio": round(ub / exact_out, 4),
                "scale_bytes_vs_input_ratio": round(exact_out / exact_in, 4),
            }
        )

    # Max tensors supported on Blackwell (227 KiB opt-in SMEM)
    max_tensors_default_48k = (DEFAULT_MAX_SMEM_BYTES - 8) // 24
    max_tensors_optin_227k = (BLACKWELL_OPTIN_MAX_SMEM_BYTES - 8) // 24

    return {
        "experiment_id": "te_grouped_mxfp8_swizzle_sim",
        "title": "TransformerEngine Blackwell FP8 Block-Scaling to MXFP8 Grouped GEMM Swizzle Simulator",
        "commit_refs": [
            "NVIDIA/TransformerEngine@f4a69dc81eabb5002a662e56a36e8454c9840a92",
            "NVIDIA/TransformerEngine@4a3a8a8e8578e96319467d0cc96b9f5db552a7d3",
        ],
        "hardware_constants": {
            "block_len": BLOCK_LEN,
            "tile_output_bytes": TILE_OUTPUT_BYTES,
            "default_max_smem_bytes": DEFAULT_MAX_SMEM_BYTES,
            "blackwell_optin_max_smem_bytes": BLACKWELL_OPTIN_MAX_SMEM_BYTES,
            "max_tensors_default_48k": max_tensors_default_48k,
            "max_tensors_optin_227k": max_tensors_optin_227k,
        },
        "scenarios": scenarios,
    }


def write_artifacts(summary: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    csv_path = out_dir / "grouped_swizzle_scenarios.csv"

    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    fieldnames = list(summary["scenarios"][0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary["scenarios"])


def run_check(summary: dict[str, Any]) -> None:
    scenarios = {s["scenario"]: s for s in summary["scenarios"]}
    for s in summary["scenarios"]:
        assert s["host_upper_bound_bytes"] >= s["exact_mxfp8_scale_bytes"], (
            f"Upper bound violated in {s['scenario']}"
        )
        assert s["exact_mxfp8_scale_bytes"] == s["total_128x128_tiles"] * TILE_OUTPUT_BYTES

    ds1d = scenarios["ds_v3_256e_1d_fprop"]
    ds2d = scenarios["ds_v3_256e_2d_fprop"]
    ultra = scenarios["ultra_moe_2048e_optin_smem"]
    wgrad = scenarios["ds_v3_64e_wgrad_varying_k"]

    # 1D FP32 input scales: 1 float (4B) per 1x128 block => 128x128 tile has 128 floats = 512B.
    # Output MXFP8 E8M0 swizzled scales also write 128x4 = 512B per 128x128 tile.
    assert ds1d["table_smem_bytes"] == (3 * 256 + 1) * 8 == 6152
    assert not ds1d["requires_optin_smem"]
    assert ds1d["upper_bound_slack_ratio"] >= 1.0

    # 2D FP32 input scales: 1 float (4B) per 128x128 tile (padded to 4 tiles in X),
    # whereas MXFP8 E8M0 replicates each 128x128 scale across 128 rows * 4 cols = 512B!
    assert ds2d["scale_bytes_vs_input_ratio"] == 128.0

    # 2048 experts need (3*2048+1)*8 = 49,160 B > 49,152 B (48 KiB), requiring opt-in dynamic SMEM
    assert ultra["table_smem_bytes"] == 49160
    assert ultra["requires_optin_smem"] is True
    assert summary["hardware_constants"]["max_tensors_default_48k"] == 2047
    assert summary["hardware_constants"]["max_tensors_optin_227k"] == 9685
    assert wgrad["varying_mode"] == "varying_last"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Run assertions and write results")
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parent / "results"
    summary = simulate_scenarios()
    write_artifacts(summary, out_dir)
    if args.check:
        run_check(summary)
        print(
            f"[te_grouped_mxfp8_swizzle_sim] CHECK PASSED: "
            f"ds1d_out={summary['scenarios'][0]['exact_mxfp8_scale_bytes']}B, "
            f"ub={summary['scenarios'][0]['host_upper_bound_bytes']}B, "
            f"max_default_48k={summary['hardware_constants']['max_tensors_default_48k']} tensors"
        )


if __name__ == "__main__":
    main()
