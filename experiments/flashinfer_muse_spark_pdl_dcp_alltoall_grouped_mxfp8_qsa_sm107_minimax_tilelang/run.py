from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_summary() -> dict[str, Any]:
    # 1. FlashInfer #7341 (31d0a811aebe): cake_muse_spark_norm PDL route (bf16_pdl) on SM100 (B200) and SM103 (GB300)
    muse_spark_pdl_benchmarks = [
        {
            "arch": "sm_100a_b200",
            "kernel": "rmsnorm_sigmoid_gate",
            "shape": "h32_T1",
            "launch_form": "pdl",
            "baseline_bf16_us": 1.248,
            "bf16_pdl_us": 0.866,
            "speedup_x": 1.441,
        },
        {
            "arch": "sm_100a_b200",
            "kernel": "gated_residual_rmsnorm",
            "shape": "mode1_M8",
            "launch_form": "pdl",
            "baseline_bf16_us": 2.272,
            "bf16_pdl_us": 1.506,
            "speedup_x": 1.509,
        },
        {
            "arch": "sm_103a_gb300",
            "kernel": "rmsnorm_sigmoid_gate",
            "shape": "h32_T1",
            "launch_form": "pdl",
            "baseline_bf16_us": 1.235,
            "bf16_pdl_us": 0.862,
            "speedup_x": 1.433,
        },
        {
            "arch": "sm_103a_gb300",
            "kernel": "gated_residual_rmsnorm",
            "shape": "mode1_M1",
            "launch_form": "pdl",
            "baseline_bf16_us": 2.007,
            "bf16_pdl_us": 1.369,
            "speedup_x": 1.466,
        },
    ]

    # Admit index binary search simulation (bisect over disjoint spans above < rows <= at_most)
    disjoint_spans = [
        {"above": 0, "at_most": 16, "form": "pdl"},
        {"above": 16, "at_most": 256, "form": "late_pdl"},
        {"above": 256, "at_most": 2048, "form": "gap_cute_dsl_pdl"},
        {"above": 2048, "at_most": 65536, "form": "plain_borrowed"},
    ]

    # 2. FlashInfer #7340 (53feaae34c0a): cake_dcp_alltoall faster CP2/CP4 kernels over NVLink LL128 FIFO
    dcp_alltoall_benchmarks = [
        {
            "arch": "sm_103a_gb300",
            "cp_size": 2,
            "tokens": 1,
            "cache_state": "warm",
            "previous_us": 5.872,
            "this_pr_us": 4.784,
            "speedup_x": 1.23,
        },
        {
            "arch": "sm_103a_gb300",
            "cp_size": 4,
            "tokens": 1,
            "cache_state": "warm",
            "previous_us": 5.440,
            "this_pr_us": 4.480,
            "speedup_x": 1.21,
        },
        {
            "arch": "sm_100a_gb200",
            "cp_size": 2,
            "tokens": 1,
            "cache_state": "warm",
            "previous_us": 5.456,
            "this_pr_us": 4.288,
            "speedup_x": 1.27,
        },
        {
            "arch": "sm_100a_gb200",
            "cp_size": 2,
            "tokens": 1,
            "cache_state": "cold",
            "previous_us": 5.584,
            "this_pr_us": 4.192,
            "speedup_x": 1.33,
        },
    ]

    # 3. FlashInfer #7322 (5396dc3b7a95): cake_grouped_mxfp8_quantize shape-based selection (bulk-copy, row-tiled, row-2D)
    def select_mxfp8_program(b: int, m: int, k: int, elem_bytes: int = 2) -> str:
        input_bytes = b * m * k * elem_bytes
        if input_bytes <= 4 * 1024 * 1024:
            return "bulk-copy"
        if b * m >= 16384:
            return "row-tiled"
        return "row-2D"

    def mxfp8_grid(program: str, b: int, m: int, k: int) -> tuple[int, int, int]:
        blocks_per_row = (k + 31) // 32
        threads = 128
        if program != "row-tiled":
            grid_x = (blocks_per_row + threads - 1) // threads
            return (grid_x, m, b)
        rows_side_by_side = max(1, threads // blocks_per_row)
        rows_per_cta = 2 * rows_side_by_side
        grid_x = (blocks_per_row + threads - 1) // threads
        grid_y = (m + rows_per_cta - 1) // rows_per_cta
        return (grid_x, grid_y, b)

    grouped_mxfp8_shapes = [
        {
            "shape": [2, 256, 4096],
            "mask_desc": "full",
            "valid_rows": 512,
            "program": select_mxfp8_program(2, 256, 4096),
            "grid": list(mxfp8_grid(select_mxfp8_program(2, 256, 4096), 2, 256, 4096)),
            "gb300_prev_us": 3.721,
            "gb300_new_us": 3.482,
            "gb300_speedup_x": 1.0678,
            "gb200_prev_us": 3.909,
            "gb200_new_us": 3.604,
            "gb200_speedup_x": 1.0838,
        },
        {
            "shape": [3, 256, 2048],
            "mask_desc": "full",
            "valid_rows": 768,
            "program": select_mxfp8_program(3, 256, 2048),
            "grid": list(mxfp8_grid(select_mxfp8_program(3, 256, 2048), 3, 256, 2048)),
            "gb300_prev_us": 3.406,
            "gb300_new_us": 3.096,
            "gb300_speedup_x": 1.0995,
            "gb200_prev_us": 3.552,
            "gb200_new_us": 3.210,
            "gb200_speedup_x": 1.1065,
        },
        {
            "shape": [8, 512, 2048],
            "mask_desc": "full",
            "valid_rows": 4096,
            "program": select_mxfp8_program(8, 512, 2048),
            "grid": list(mxfp8_grid(select_mxfp8_program(8, 512, 2048), 8, 512, 2048)),
            "gb300_prev_us": 6.857,
            "gb300_new_us": 6.861,
            "gb300_speedup_x": 1.0003,
            "gb200_prev_us": 7.066,
            "gb200_new_us": 7.062,
            "gb200_speedup_x": 0.9999,
        },
        {
            "shape": [32, 1024, 7168],
            "mask_desc": "16_per_group",
            "valid_rows": 512,
            "program": select_mxfp8_program(32, 1024, 7168),
            "grid": list(mxfp8_grid(select_mxfp8_program(32, 1024, 7168), 32, 1024, 7168)),
            "gb300_prev_us": 34.219,
            "gb300_new_us": 18.879,
            "gb300_speedup_x": 1.8125,
            "gb200_prev_us": 35.968,
            "gb200_new_us": 19.796,
            "gb200_speedup_x": 1.8166,
        },
        {
            "shape": [32, 1024, 7168],
            "mask_desc": "512_in_group_0",
            "valid_rows": 512,
            "program": select_mxfp8_program(32, 1024, 7168),
            "grid": list(mxfp8_grid(select_mxfp8_program(32, 1024, 7168), 32, 1024, 7168)),
            "gb300_prev_us": 34.724,
            "gb300_new_us": 17.955,
            "gb300_speedup_x": 1.9340,
            "gb200_prev_us": 36.392,
            "gb200_new_us": 18.802,
            "gb200_speedup_x": 1.9357,
        },
        {
            "shape": [32, 1024, 7168],
            "mask_desc": "full",
            "valid_rows": 32768,
            "program": select_mxfp8_program(32, 1024, 7168),
            "grid": list(mxfp8_grid(select_mxfp8_program(32, 1024, 7168), 32, 1024, 7168)),
            "gb300_prev_us": 103.497,
            "gb300_new_us": 104.558,
            "gb300_speedup_x": 0.9899,
            "gb200_prev_us": 102.500,
            "gb200_new_us": 103.954,
            "gb200_speedup_x": 0.9860,
        },
    ]

    # 4. FlashInfer #7284 (cc8552a3a015): cake_qsa_nvfp4 tcgen05 decode on Rubin SM107 (R200, 212 SMs)
    def qsa_num_splits(rows: int, num_kv_heads: int, topk: int, sm_count: int, ctas_per_sm: int = 1) -> int:
        tiles = (topk + 63) // 64
        capacity = ctas_per_sm * sm_count // (rows * num_kv_heads)
        return max(1, min(tiles, capacity, 64))

    r200_rows = [1, 2, 4, 8, 16, 32, 64, 128]
    r200_splits_212 = [qsa_num_splits(r, 2, 2051, 212, 1) for r in r200_rows]
    r200_qsa_nhd_eager = [
        {"rows": 1, "splits": r200_splits_212[0], "ref_us": 13.15, "cake_us": 9.15, "speedup_x": 1.44},
        {"rows": 2, "splits": r200_splits_212[1], "ref_us": 13.87, "cake_us": 9.54, "speedup_x": 1.45},
        {"rows": 4, "splits": r200_splits_212[2], "ref_us": 13.81, "cake_us": 10.37, "speedup_x": 1.33},
        {"rows": 8, "splits": r200_splits_212[3], "ref_us": 16.98, "cake_us": 11.18, "speedup_x": 1.52},
        {"rows": 16, "splits": r200_splits_212[4], "ref_us": 24.11, "cake_us": 13.42, "speedup_x": 1.80},
        {"rows": 32, "splits": r200_splits_212[5], "ref_us": 39.12, "cake_us": 16.59, "speedup_x": 2.36},
        {"rows": 64, "splits": r200_splits_212[6], "ref_us": 68.56, "cake_us": 29.07, "speedup_x": 2.36},
        {"rows": 128, "splits": r200_splits_212[7], "ref_us": 151.44, "cake_us": 55.52, "speedup_x": 2.73},
    ]

    # 5. TileLang #3503 (fe6d319c78c0) & #2342 (fc0fbf0e50de)
    tilelang_verification = {
        "ws_tma_producer_barrier_fix": {
            "commit": "fe6d319c78c0738186fc6e83eb8436ddef65204d",
            "issue": "#3470 / #3503",
            "bug_mechanism": "Pure-TMA producer with multiple warps or prefix bindings only had elected thread arrive on forward mbarrier, leaving warps 1-3 waiting only on back-pressure barriers and falling 2 phases behind",
            "producer_is_single_warp_formula": "!has_async_producer && !has_producer_prefix && !loop_body_condition.defined()",
            "single_warp_fwd_arrive_count": 1,
            "multi_warp_or_simt_fwd_arrive_count": 128,
        },
        "sibling_pipeline_buffer_reconciler": {
            "commit": "fc0fbf0e50de55c8e3f04b4954c2ab1000b8a298",
            "issue": "#2309 / #2342",
            "unfixed_double_expansion_dims": [4, 2, 64, 32],
            "reconciled_max_stage_dims": [4, 64, 32],
            "tested_stage_pairs": [[2, 4], [4, 2], [3, 1], [1, 3], [3, 3]],
        },
    }

    return {
        "experiment_id": "flashinfer_muse_spark_pdl_dcp_alltoall_grouped_mxfp8_qsa_sm107_minimax_tilelang",
        "phase": 79,
        "priority": 88,
        "evidence_mode": "own-offline-model-plus-author-benchmark-table",
        "muse_spark_pdl": {
            "commit": "31d0a811aebed9047ee60d27760d7872600ad93f",
            "disjoint_admit_spans": disjoint_spans,
            "benchmarks": muse_spark_pdl_benchmarks,
            "max_speedup_b200_x": 1.509,
            "max_speedup_gb300_x": 1.466,
        },
        "dcp_alltoall_ll128": {
            "commit": "53feaae34c0a9c7bd8801676a150c4321b496c16",
            "mechanism": "deferred_last_fifo_slot_release_at_next_launch_plus_register_plain_stores",
            "benchmarks": dcp_alltoall_benchmarks,
            "speedup_range_x": [1.15, 1.33],
        },
        "grouped_mxfp8_quantize": {
            "commit": "5396dc3b7a955386c37a606c97c8269f8bab9a73",
            "bulk_threshold_bytes": 4 * 1024 * 1024,
            "row_tiled_threshold_rows": 16384,
            "shapes": grouped_mxfp8_shapes,
            "max_sparse_speedup_gb200_x": 1.9357,
            "max_sparse_speedup_gb300_x": 1.9340,
        },
        "qsa_nvfp4_sm107_r200": {
            "commit": "cc8552a3a0158391f514571597b3b495090ce43c",
            "arch": "sm_107a",
            "sm_count": 212,
            "ctas_per_sm": 1,
            "splits_212_sms": r200_splits_212,
            "nhd_eager_benchmarks": r200_qsa_nhd_eager,
            "max_speedup_eager_memset_x": 2.73,
            "max_speedup_eager_read_x": 2.76,
        },
        "sm120_minimax_h3": {
            "varlen_attn_commit": "b8ed186df31b1ad175d6b0b4a47120542c48279d",
            "out_proj_commit": "53ecbde224f45741a2aa4a0b928a330a502ab15e",
            "rtx5090_attn_speedup_range_x": [1.014, 1.055],
            "rtx_pro_6000_attn_speedup_range_x": [1.030, 1.041],
            "out_proj_speedup_range_x": [1.006, 1.011],
        },
        "tilelang": tilelang_verification,
    }


def verify_summary(summary: dict[str, Any]) -> None:
    assert summary["phase"] == 79
    assert summary["priority"] == 88
    assert summary["muse_spark_pdl"]["max_speedup_b200_x"] == 1.509
    assert summary["muse_spark_pdl"]["max_speedup_gb300_x"] == 1.466
    assert summary["dcp_alltoall_ll128"]["speedup_range_x"] == [1.15, 1.33]
    shapes = summary["grouped_mxfp8_quantize"]["shapes"]
    assert shapes[0]["program"] == "bulk-copy"
    assert shapes[2]["program"] == "row-2D"
    assert shapes[3]["program"] == "row-tiled"
    assert shapes[3]["grid"] == [2, 512, 32]
    assert summary["qsa_nvfp4_sm107_r200"]["splits_212_sms"] == [33, 33, 26, 13, 6, 3, 1, 1]
    assert summary["qsa_nvfp4_sm107_r200"]["max_speedup_eager_memset_x"] == 2.73
    assert summary["tilelang"]["sibling_pipeline_buffer_reconciler"]["reconciled_max_stage_dims"] == [4, 64, 32]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    out_path = Path(__file__).with_name("summary.json")
    summary = build_summary()
    verify_summary(summary)

    if args.check:
        if out_path.exists():
            existing = json.loads(out_path.read_text(encoding="utf-8"))
            verify_summary(existing)
        print("OK: flashinfer_muse_spark_pdl_dcp_alltoall_grouped_mxfp8_qsa_sm107_minimax_tilelang verified")
        return

    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
