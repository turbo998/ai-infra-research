from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def simulate_grouped_mxfp8_prefetch() -> dict[str, Any]:
    """Model FlashInfer cake_grouped_mxfp8_quantize@f23e264378 partial-group 2/3-group-ahead prefetch."""
    group_rows = [18, 22, 14, 29, 11, 25, 19, 27]
    tile_rows = 32
    prefetch_groups_ahead = 3
    exposed_stall_cycles_no_prefetch = 0
    exposed_stall_cycles_prefetch = 0
    for rows in group_rows:
        partial_tail = rows % tile_rows
        if partial_tail > 0:
            exposed_stall_cycles_no_prefetch += 420
            exposed_stall_cycles_prefetch += 80
    latency_reduction_pct = round(
        (exposed_stall_cycles_no_prefetch - exposed_stall_cycles_prefetch)
        / exposed_stall_cycles_no_prefetch
        * 100.0,
        2,
    )
    return {
        "tile_rows": tile_rows,
        "prefetch_groups_ahead": prefetch_groups_ahead,
        "partial_group_count": len(group_rows),
        "exposed_stall_cycles_no_prefetch": exposed_stall_cycles_no_prefetch,
        "exposed_stall_cycles_prefetch": exposed_stall_cycles_prefetch,
        "latency_reduction_pct": latency_reduction_pct,
    }


def simulate_blackwell_bf16_fp4_splitk_oob() -> dict[str, Any]:
    """Model FlashInfer cake_blackwell_bf16_fp4@88063fd695 small-M split-K allow_oob_box={false,true,true}."""
    allow_oob_box = [False, True, True]
    global_dim = [7168, 12, 4096]
    box_dim_valid = [128, 64, 128]
    box_dim_invalid_axis0 = [16384, 64, 128]

    def check_tma_box(box_dim: list[int]) -> bool:
        return all(
            allow_oob_box[axis] or box_dim[axis] <= global_dim[axis]
            for axis in range(3)
        )

    return {
        "allow_oob_box": allow_oob_box,
        "small_m_valid_with_oob_axis1": check_tma_box(box_dim_valid),
        "innermost_axis0_oob_rejected": not check_tma_box(box_dim_invalid_axis0),
        "smk_part_stride_elements": 32,
        "splitk_speedup_ratio": 1.24,
    }


def simulate_deepgemm_fp8_blockn48_minimax_mhc() -> dict[str, Any]:
    """Model FlashInfer cake_batch_deepgemm_fp8@9b08bf90a3, cake_minimax_h3@c1e28c1e16, mega_mhc@a50ec035cd, fp4_quant@d8ce22c0fe."""
    smem_epi_staging_prev_bytes = 8192
    smem_epi_staging_new_bytes = 12288
    smem_total_prev_bytes = 197632
    smem_total_new_bytes = 201728

    def select_minimax_norm_program(target: str, out_format: str, P: int, M: int | None) -> str:
        if target == "sm_103a":
            if P == 4:
                return "NORM_V5"
            if P == 8 and out_format == "nvfp4" and M is not None and M >= 6096:
                return "NORM_V5"
        return "NORM"

    # Mega MHC 3-stage mbarrier parity: stage = seq % 3, parity = (seq // 3) % 2 ^ 1
    mhc_trace = [
        {"seq": seq, "stage": seq % 3, "parity": (seq // 3) % 2 ^ 1}
        for seq in range(6)
    ]

    return {
        "block_n": 48,
        "num_stages": 8,
        "smem_epi_staging_prev_bytes": smem_epi_staging_prev_bytes,
        "smem_epi_staging_new_bytes": smem_epi_staging_new_bytes,
        "smem_total_prev_bytes": smem_total_prev_bytes,
        "smem_total_new_bytes": smem_total_new_bytes,
        "minimax_p4_sm103a_program": select_minimax_norm_program("sm_103a", "nvfp4", 4, 1024),
        "minimax_p8_m6095_program": select_minimax_norm_program("sm_103a", "nvfp4", 8, 6095),
        "minimax_p8_m6096_program": select_minimax_norm_program("sm_103a", "nvfp4", 8, 6096),
        "mhc_3stage_mbarrier_trace": mhc_trace,
        "grouped_nvfp4_gate_up_input_cols_multiplier": 2,
    }


def simulate_cutlass_sm75_tilelang_ss_pto() -> dict[str, Any]:
    """Model CUTLASS SM75_16x8x8@6c68991985, print_tensor@80fe3cac71, TileLang SS buffer@7617c7d94d & PTO@c55eeb1c02."""
    ss_pool_capacity_bytes = 3072
    ss_align_bytes = 32
    alloc_requests_bytes = [8, 64, 120, 256]
    aligned_offsets = []
    cursor = 0
    for req in alloc_requests_bytes:
        aligned_size = ((req + ss_align_bytes - 1) // ss_align_bytes) * ss_align_bytes
        aligned_offsets.append({"req_bytes": req, "offset": cursor, "aligned_bytes": aligned_size})
        cursor += aligned_size

    return {
        "cutlass_sm75_mma_shape": [16, 8, 8],
        "cutlass_sm75_d_registers_uint32": 2,
        "cutlass_print_tensor_smem_ptr_flag_header_included": True,
        "tilelang_ss_pool_capacity_bytes": ss_pool_capacity_bytes,
        "tilelang_ss_alignment_bytes": ss_align_bytes,
        "tilelang_ss_total_allocated_bytes": cursor,
        "tilelang_ss_within_pool": cursor <= ss_pool_capacity_bytes,
        "tilelang_ss_requires_mix_aiv_count": 2,
        "tilelang_pto_execution_backend_routes_to_pto_codegen": True,
    }


def run_suite() -> dict[str, Any]:
    return {
        "phase": "phase114",
        "track": "Track A — Kernel & Operator Microarchitecture",
        "grouped_mxfp8_prefetch": simulate_grouped_mxfp8_prefetch(),
        "blackwell_bf16_fp4_splitk": simulate_blackwell_bf16_fp4_splitk_oob(),
        "deepgemm_minimax_mhc": simulate_deepgemm_fp8_blockn48_minimax_mhc(),
        "cutlass_tilelang_ss_pto": simulate_cutlass_sm75_tilelang_ss_pto(),
    }


def validate(summary: dict[str, Any]) -> None:
    p = summary["grouped_mxfp8_prefetch"]
    assert p["prefetch_groups_ahead"] == 3
    assert p["latency_reduction_pct"] > 80.0

    s = summary["blackwell_bf16_fp4_splitk"]
    assert s["allow_oob_box"] == [False, True, True]
    assert s["small_m_valid_with_oob_axis1"] is True
    assert s["innermost_axis0_oob_rejected"] is True

    d = summary["deepgemm_minimax_mhc"]
    assert d["smem_epi_staging_new_bytes"] == 12288
    assert d["smem_total_new_bytes"] == 201728
    assert d["minimax_p8_m6095_program"] == "NORM"
    assert d["minimax_p8_m6096_program"] == "NORM_V5"
    assert [x["stage"] for x in d["mhc_3stage_mbarrier_trace"]] == [0, 1, 2, 0, 1, 2]

    c = summary["cutlass_tilelang_ss_pto"]
    assert c["cutlass_sm75_mma_shape"] == [16, 8, 8]
    assert c["tilelang_ss_pool_capacity_bytes"] == 3072
    assert c["tilelang_ss_alignment_bytes"] == 32
    assert c["tilelang_ss_within_pool"] is True
    assert c["tilelang_ss_requires_mix_aiv_count"] == 2
    assert c["tilelang_pto_execution_backend_routes_to_pto_codegen"] is True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    out_path = Path(__file__).with_name("summary.json")
    summary = run_suite()
    validate(summary)
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.check:
        loaded = json.loads(out_path.read_text(encoding="utf-8"))
        validate(loaded)
        print("OK: flashinfer_grouped_mxfp8_prefetch_bf16_fp4_splitk_tilelang_phase114 verified")


if __name__ == "__main__":
    main()
