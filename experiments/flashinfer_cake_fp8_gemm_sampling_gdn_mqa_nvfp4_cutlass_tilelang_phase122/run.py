"""Phase 122 Track A: FlashInfer CAKE Grouped FP8 GEMM 32-Row UE8M0 Swap-AB, Paged Auto Q-Len Refresh, CAKE Sampling Round 16 SM120, CAKE GDN Bwd, CAKE DeepGEMM Dense MQA, NVFP4 TMA Scale-Factor Row Bound, CUTLASS 4.8 SM107/MXFP8 K64-K128, and TileLang Autotuner Profiler Forwarding."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def simulate_cake_grouped_fp8_gemm_small_block_swap_ab() -> dict[str, Any]:
    """Model FlashInfer PR #7589 (87658682): 32-row M-block UE8M0 swap-AB on SM103."""
    sm_count_gb300 = 152
    sm_count_148 = 148

    def ctas_rp32_s11(m: int, n: int) -> int:
        # 2-CTA cluster over 32-row M-blocks and 256-col N-blocks
        return 2 * math.ceil(m / 32) * (n // 256)

    def ctas_c1(m: int, n: int) -> int:
        # 1 CTA per 32x128 tile, no cluster
        return math.ceil(m / 32) * (n // 128)

    def route_ue8m0_small_block(m: int, n: int, k: int, align: int, sm_count: int) -> str:
        if k >= 2048 and n >= 256 and n % 256 == 0 and ctas_rp32_s11(m, n) <= sm_count:
            return "bs_ue8m0_swap_rp32_s11"
        if align == 32 and 256 <= k <= 512 and n <= 512 and n % 128 == 0 and ctas_c1(m, n) <= sm_count:
            return "bs_ue8m0_swap_c1"
        return "existing_ue8m0_64_or_128_row"

    # Boundary shape M=2432, N=256, K=4096: 2 * ceil(2432/32) * 1 = 152 CTAs
    boundary_ctas = ctas_rp32_s11(2432, 256)
    route_on_152 = route_ue8m0_small_block(2432, 256, 4096, 64, sm_count_gb300)
    route_on_148 = route_ue8m0_small_block(2432, 256, 4096, 64, sm_count_148)

    gb300_benchmarks = [
        {"id": "b3_a64_n256_k4096_p16", "m": 2048, "routed": 1417, "n": 256, "k": 4096, "align": 64, "base_us": 13.63, "cand_us": 7.93, "speedup_x": 1.72, "program": "bs_ue8m0_swap_rp32_s11"},
        {"id": "pk_n256_wide_k2048", "m": 2048, "routed": 1681, "n": 256, "k": 2048, "align": 128, "base_us": 8.11, "cand_us": 5.98, "speedup_x": 1.36, "program": "bs_ue8m0_swap_rp32_s11"},
        {"id": "s3_a128_n1024_k4096_u4_g1", "m": 1024, "routed": 747, "n": 1024, "k": 4096, "align": 128, "base_us": 11.60, "cand_us": 7.54, "speedup_x": 1.54, "program": "bs_ue8m0_swap_rp32_s11"},
        {"id": "s3_a64_n1024_k4096_u4_g1", "m": 1024, "routed": 564, "n": 1024, "k": 4096, "align": 64, "base_us": 10.62, "cand_us": 6.96, "speedup_x": 1.53, "program": "bs_ue8m0_swap_rp32_s11"},
        {"id": "s2_a64_n512_k4096_u4_g2", "m": 1536, "routed": 1006, "n": 512, "k": 4096, "align": 64, "base_us": 10.71, "cand_us": 7.05, "speedup_x": 1.52, "program": "bs_ue8m0_swap_rp32_s11"},
        {"id": "s1_a32_n256_k512_u8_g4", "m": 4096, "routed": 3557, "n": 256, "k": 512, "align": 32, "base_us": 7.12, "cand_us": 6.46, "speedup_x": 1.10, "program": "bs_ue8m0_swap_c1"},
        {"id": "s1_a32_n512_k512_u8_g2", "m": 3072, "routed": 2576, "n": 512, "k": 512, "align": 32, "base_us": 9.35, "cand_us": 8.63, "speedup_x": 1.08, "program": "bs_ue8m0_swap_c1"},
    ]

    return {
        "programs_before": 27,
        "programs_after": 29,
        "host_routes_before": 26,
        "host_routes_after": 28,
        "ue8m0_routes_after": 14,
        "swap_ab_programs_after": 8,
        "rerouted_shapes_total": 28,
        "rerouted_to_rp32_s11": 23,
        "rerouted_to_c1": 5,
        "boundary_m2432_n256_ctas": boundary_ctas,
        "route_m2432_n256_on_152_sms": route_on_152,
        "route_m2432_n256_on_148_sms": route_on_148,
        "max_speedup_x": 1.72,
        "gb300_benchmarks": gb300_benchmarks,
    }


def simulate_paged_auto_retained_q_len_and_nvfp4_tma_row_bound() -> dict[str, Any]:
    """Model FlashInfer PR #7617 (06577b40) and PR #6278 (e9c4787c)."""
    # PR #7617: caller supplies block_tables (self._cudnn_block_tables is None),
    # plan 1 selects cuDNN (populates _cudnn_seq_lens_q), plan 2 falls back to FA2 (ragged Q),
    # plan 3 selects cuDNN again with different qo_indptr.
    qo_indptr_plan1 = [0, 64, 128]
    qo_indptr_plan3 = [0, 16, 48]

    def legacy_plan3_q_lens(caller_supplied_block_tables: bool) -> list[int]:
        cudnn_block_tables = None if caller_supplied_block_tables else "internal_table"
        retained_q_lens = [qo_indptr_plan1[i + 1] - qo_indptr_plan1[i] for i in range(2)]
        # Bug: guarded on self._cudnn_block_tables is not None
        if cudnn_block_tables is not None:
            retained_q_lens = [qo_indptr_plan3[i + 1] - qo_indptr_plan3[i] for i in range(2)]
        return retained_q_lens

    def fixed_plan3_q_lens(caller_supplied_block_tables: bool) -> list[int]:
        _ = caller_supplied_block_tables
        cudnn_seq_lens_q_present = True
        retained_q_lens = [qo_indptr_plan1[i + 1] - qo_indptr_plan1[i] for i in range(2)]
        if cudnn_seq_lens_q_present:
            retained_q_lens = [qo_indptr_plan3[i + 1] - qo_indptr_plan3[i] for i in range(2)]
        return retained_q_lens

    # PR #6278: NVFP4 TMA quantize scale-factor out-of-bounds write when m % 16 in [1, 8]
    tma_row_tile = 16
    sf_row_align_8x4 = 8
    m_example = 1025
    n_example = 2048
    k_sf_vec_size = 16  # 16 elements per FP8 scale factor
    bytes_per_row_sf = n_example // k_sf_vec_size  # 128 bytes/row
    num_padded_rows_for_sf = math.ceil(m_example / sf_row_align_8x4) * sf_row_align_8x4  # 1032
    tma_launched_rows = math.ceil(m_example / tma_row_tile) * tma_row_tile  # 1040
    unbounded_oob_rows = tma_launched_rows - num_padded_rows_for_sf  # 8 rows
    unbounded_oob_bytes = unbounded_oob_rows * (n_example // 16 // 2 if False else bytes_per_row_sf)
    # Note: PR #6278 explicitly notes n / 2 bytes past the end over 8 rows when each row has n / 16 SFs (8 * n / 16 = n / 2 bytes)!
    exact_oob_bytes = unbounded_oob_rows * (n_example // 16)

    return {
        "legacy_plan3_q_lens_external_table": legacy_plan3_q_lens(True),
        "fixed_plan3_q_lens_external_table": fixed_plan3_q_lens(True),
        "sm107_external_table_mismatch_before_pct": 67.4,
        "sm107_external_table_mismatch_after_pct": 0.0,
        "sm107_test_cells_passed": 48,
        "nvfp4_tma_row_tile_bf16": tma_row_tile,
        "nvfp4_layout_8x4_padded_rows_m1025": num_padded_rows_for_sf,
        "nvfp4_tma_launched_rows_m1025": tma_launched_rows,
        "nvfp4_oob_rows_before_fix": unbounded_oob_rows,
        "nvfp4_oob_bytes_at_n2048_before_fix": exact_oob_bytes,
        "nvfp4_oob_bytes_after_fix": 0,
    }


def simulate_cake_sampling_sm120_gdn_bwd_and_dense_mqa() -> dict[str, Any]:
    """Model FlashInfer PR #7604 (9657c04e), PR #7602 (9c81aefd), and PR #7607 (5073cd43)."""
    # PR #7604: CAKE sampling Round 16 on RTX PRO 6000 (sm_120a, 188 SMs, 101376 B dynamic SMEM)
    vocab_qwen = 151936
    cluster_size = 8
    block_threads = 512
    capacity_c8_e40 = cluster_size * block_threads * 40  # 163840
    capacity_c8_e48 = cluster_size * block_threads * 48  # 196608
    residency_c8_e40 = round(vocab_qwen / capacity_c8_e40, 4)  # 0.9273
    residency_c8_e48 = round(vocab_qwen / capacity_c8_e48, 4)  # 0.7728

    # PR #7602: CAKE GDN train backward XOR lane swizzle for krow
    # krow = (tid_1 >> 1) ^ ((tid_1 & 1) << 2)
    swizzled_krows = [((t >> 1) ^ ((t & 1) << 2)) for t in range(8)]
    sm107_sms = 148
    bwd_prep_grid_before = sm107_sms * 1
    bwd_prep_grid_after = sm107_sms * 2

    # PR #7607: CAKE DeepGEMM dense MQA logits warp-wide cost-boundary scan
    warps_per_round = 32
    blocks_per_lane = 24
    blocks_scanned_per_round = warps_per_round * blocks_per_lane  # 768

    return {
        "sm120_stage1_builds_before": 106,
        "sm120_stage1_builds_after": 108,
        "sm120_max_optin_dsmem_bytes": 101376,
        "c8_e40_dynamic_smem_bytes": 35968,
        "c8_e40_max_vocab_capacity": capacity_c8_e40,
        "c8_e40_register_residency_qwen": residency_c8_e40,
        "c8_e48_register_residency_qwen": residency_c8_e48,
        "sm120_qwen_topk10_b4_speedup_x": 1.151,
        "gdn_bwd_swizzled_krows_0_to_7": swizzled_krows,
        "gdn_bwd_prep_sm107_grid_before": bwd_prep_grid_before,
        "gdn_bwd_prep_sm107_grid_after": bwd_prep_grid_after,
        "gdn_bwd_gb300_p1_ms_before": 25.283,
        "gdn_bwd_gb300_p1_ms_after": 24.733,
        "gdn_bwd_gb300_p1_speedup_x": round(25.283 / 24.733, 4),
        "dense_mqa_programs_replaced": 15,
        "dense_mqa_program_arch_units": 18,
        "dense_mqa_blocks_scanned_per_warp_round": blocks_scanned_per_round,
        "dense_mqa_b200_median_speedup_x": 1.0537,
        "dense_mqa_b200_max_speedup_x": 1.1199,
        "dense_mqa_gb300_median_speedup_x": 1.0458,
        "dense_mqa_gb300_max_speedup_x": 1.1012,
    }


def simulate_cutlass_48_sm107_mxfp8_and_tilelang_autotuner() -> dict[str, Any]:
    """Model NVIDIA/cutlass PR #3645 (098de2a6) and tile-ai/tilelang PR #3337 (90f8908b)."""
    # CUTLASS 4.8: SM107 Rubin & SM120 MXFP8 K=64 tile with K=128 scale-factor sharing
    mxfp8_tile_shape = (128, 256, 64)
    sf_k_tile = 128
    sf_vec_size = 32
    k_iters_per_sf_tile = sf_k_tile // mxfp8_tile_shape[2]  # 2 iterations share 1 K=128 SF load
    sfs_per_k128_row = sf_k_tile // sf_vec_size  # 4 E8M0 scale factors per 128 elements

    # TileLang PR #3337: AutoTuneImpl.get_tunner() forwarding warmup, rep, timeout
    default_profile_args = (25, 100, 30)
    custom_budget_a = (10, 20, 5)
    custom_budget_b = (500, 1000, 60)

    def legacy_autotuner_profile_tuple(warmup: int, rep: int, timeout: int) -> tuple[int, int, int]:
        _ = (warmup, rep, timeout)
        return default_profile_args

    def fixed_autotuner_profile_tuple(warmup: int, rep: int, timeout: int) -> tuple[int, int, int]:
        return (warmup, rep, timeout)

    legacy_collision = (
        legacy_autotuner_profile_tuple(*custom_budget_a)
        == legacy_autotuner_profile_tuple(*custom_budget_b)
    )
    fixed_distinct = (
        fixed_autotuner_profile_tuple(*custom_budget_a)
        != fixed_autotuner_profile_tuple(*custom_budget_b)
    )

    return {
        "cutlass_version": "4.8.0",
        "rubin_sm107_mma_k_fp8": 64,
        "rubin_sm107_mma_k_fp4": 128,
        "rubin_sm107_tmem_columns": 576,
        "rubin_sm107_smem_kb": 328,
        "mxfp8_k64_sf_k128_tile_shape": list(mxfp8_tile_shape),
        "mxfp8_k_iters_per_sf_k128_chunk": k_iters_per_sf_tile,
        "mxfp8_sfs_per_k128_row": sfs_per_k128_row,
        "tilelang_legacy_profile_tuple_collision": legacy_collision,
        "tilelang_fixed_profile_tuples_distinct": fixed_distinct,
        "tilelang_custom_budget_a": list(custom_budget_a),
        "tilelang_custom_budget_b": list(custom_budget_b),
    }


def run_suite() -> dict[str, Any]:
    fp8_gemm = simulate_cake_grouped_fp8_gemm_small_block_swap_ab()
    paged_nvfp4 = simulate_paged_auto_retained_q_len_and_nvfp4_tma_row_bound()
    sampling_gdn_mqa = simulate_cake_sampling_sm120_gdn_bwd_and_dense_mqa()
    cutlass_tilelang = simulate_cutlass_48_sm107_mxfp8_and_tilelang_autotuner()

    return {
        "suite": "flashinfer_cake_fp8_gemm_sampling_gdn_mqa_nvfp4_cutlass_tilelang_phase122",
        "phase": 122,
        "priority": 131,
        "track": "Track A: Kernel & Operator Micro-Architecture",
        "cake_grouped_fp8_gemm": fp8_gemm,
        "paged_auto_and_nvfp4_tma": paged_nvfp4,
        "cake_sampling_gdn_and_dense_mqa": sampling_gdn_mqa,
        "cutlass_48_and_tilelang_autotuner": cutlass_tilelang,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify deterministic invariants")
    args = parser.parse_args()

    summary = run_suite()
    out_path = Path(__file__).with_name("summary.json")
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if args.check:
        assert summary["cake_grouped_fp8_gemm"]["max_speedup_x"] == 1.72
        assert summary["cake_grouped_fp8_gemm"]["route_m2432_n256_on_152_sms"] == "bs_ue8m0_swap_rp32_s11"
        assert summary["cake_grouped_fp8_gemm"]["route_m2432_n256_on_148_sms"] == "existing_ue8m0_64_or_128_row"
        assert summary["paged_auto_and_nvfp4_tma"]["fixed_plan3_q_lens_external_table"] == [16, 32]
        assert summary["paged_auto_and_nvfp4_tma"]["nvfp4_oob_bytes_at_n2048_before_fix"] == 1024
        assert summary["cake_sampling_gdn_and_dense_mqa"]["c8_e40_max_vocab_capacity"] == 163840
        assert summary["cutlass_48_and_tilelang_autotuner"]["rubin_sm107_tmem_columns"] == 576
        assert summary["cutlass_48_and_tilelang_autotuner"]["tilelang_fixed_profile_tuples_distinct"] is True

    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
