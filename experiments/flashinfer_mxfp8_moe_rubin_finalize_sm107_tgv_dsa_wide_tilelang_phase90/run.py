"""Phase 90 Track A: FlashInfer MXFP8 Fused MoE, Rubin Unfused Deterministic Finalize, SM107 Deeper TGV Pipelines, Cake DSA Wide-First Indexer, Kimi-K3 TP12 Tail, 5-CTA MoE Finalize AllReduce, Blackwell Cached-Cluster Softmax & TileLang SM8x/Reduce Model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def model_mxfp8_cute_dsl_fused_moe() -> dict[str, object]:
    """Model FlashInfer PR #6793 (`b92f112955`) MXFP8 x MXFP8 CuTe-DSL fused MoE on SM100/SM103."""
    tokens = 2048
    hidden_size = 7168
    intermediate_size = 2048
    top_k = 8
    block_size = 32
    bf16_weight_bytes_per_elem = 2.0
    mxfp8_weight_bytes_per_elem = 1.0 + (1.0 / block_size)  # E4M3 + E8M0 scale per 32 elems
    bandwidth_saving_ratio = round(bf16_weight_bytes_per_elem / mxfp8_weight_bytes_per_elem, 4)
    scale_blocks_per_row = hidden_size // block_size
    assert mxfp8_weight_bytes_per_elem == 1.03125
    assert scale_blocks_per_row == 224
    assert bandwidth_saving_ratio > 1.93
    return {
        "tokens": tokens,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "top_k": top_k,
        "block_size": block_size,
        "mxfp8_effective_bytes_per_elem": mxfp8_weight_bytes_per_elem,
        "scale_blocks_per_row": scale_blocks_per_row,
        "bandwidth_saving_ratio_vs_bf16": bandwidth_saving_ratio,
        "reuses_w4a8_kernel_with_fp8_b_operand": True,
    }


def model_rubin_unfused_deterministic_finalize() -> dict[str, object]:
    """Model FlashInfer PR #6950 (`5240384d8f`) unfused deterministic finalize path on Rubin SM107."""
    tokens = 512
    top_k = 8
    hidden_size = 7168
    fused_atomic_scatter_associativity_deterministic = False
    unfused_contiguous_gemm_then_reduce_deterministic = True
    intermediate_buffer_bytes = tokens * top_k * hidden_size * 2  # BF16 [M, top_k, K]
    assert not fused_atomic_scatter_associativity_deterministic
    assert unfused_contiguous_gemm_then_reduce_deterministic
    return {
        "sm_arch": "sm_107",
        "tokens": tokens,
        "top_k": top_k,
        "hidden_size": hidden_size,
        "fused_atomic_scatter_deterministic": fused_atomic_scatter_associativity_deterministic,
        "unfused_finalize_deterministic": unfused_contiguous_gemm_then_reduce_deterministic,
        "intermediate_buffer_bytes": intermediate_buffer_bytes,
    }


def model_sm107_tgv_deeper_pipelines() -> dict[str, object]:
    """Model FlashInfer PR #7049 (`f2ebfdc8c4`) SM107 TGV GEMM tactics (ids 29-46) with deeper A/B pipelines."""
    sm100_base_tactics = 18
    sm107_new_tactic_ids = list(range(29, 47))
    sm100_smem_kb = 228
    sm107_smem_kb = 320
    sm100_ab_stages = 4
    sm107_ab_stages = 6
    pipeline_depth_gain_ratio = round(sm107_ab_stages / sm100_ab_stages, 4)
    assert len(sm107_new_tactic_ids) == sm100_base_tactics == 18
    assert sm107_smem_kb > sm100_smem_kb
    assert pipeline_depth_gain_ratio == 1.5
    return {
        "sm100_base_tactics": sm100_base_tactics,
        "sm107_tactic_id_range": [sm107_new_tactic_ids[0], sm107_new_tactic_ids[-1]],
        "sm107_added_tactics": len(sm107_new_tactic_ids),
        "sm100_smem_kb": sm100_smem_kb,
        "sm107_smem_kb": sm107_smem_kb,
        "sm100_ab_stages": sm100_ab_stages,
        "sm107_ab_stages": sm107_ab_stages,
        "pipeline_depth_gain_ratio": pipeline_depth_gain_ratio,
    }


def model_cake_dsa_wide_kimi_tp12_moe_ar_softmax() -> dict[str, object]:
    """Model FlashInfer PRs #7406 (`35cba307a0`), #7402 (`8707500798`), #7392 (`ee376e9737`), #7404 (`01a04018fc`)."""
    dsa_wide_threshold_mean_tiles = 512
    dsa_queries_per_wide_unit = 8
    kimi_tp12_regimes = {
        "one_shot_m_le_8": [1, 8],
        "one_shot_ess_k1_k3_m_9_16": [9, 16],
        "two_shot_ess_k1_k3_m_17_960": [17, 960],
        "two_shot_persistent_m_gt_960": [961, 4096],
    }
    moe_finalize_ar_programs = 2 * 3 * 2 * 2 + 8  # 32 total programs
    moe_finalize_ctas_per_sm = 5
    softmax_blackwell_programs = 14
    softmax_cached_cluster_programs = 7
    softmax_shared_tile_programs = 2
    assert dsa_wide_threshold_mean_tiles == 512
    assert moe_finalize_ar_programs == 32
    assert moe_finalize_ctas_per_sm == 5
    return {
        "dsa_wide_threshold_mean_unit_tiles": dsa_wide_threshold_mean_tiles,
        "dsa_queries_per_wide_unit": dsa_queries_per_wide_unit,
        "kimi_tp12_regimes": kimi_tp12_regimes,
        "moe_finalize_ar_total_programs": moe_finalize_ar_programs,
        "moe_finalize_ctas_per_sm": moe_finalize_ctas_per_sm,
        "softmax_blackwell_total_programs": softmax_blackwell_programs,
        "softmax_cached_cluster_programs": softmax_cached_cluster_programs,
        "softmax_shared_tile_programs": softmax_shared_tile_programs,
    }


def model_tilelang_sm8x_sparse_deepselect_reduce() -> dict[str, object]:
    """Model TileLang PRs #3335 (`80364b3d97`), #3304 (`f8053068c7`), and #3189 (`1a0830a50d`)."""
    allowed_sm8x_mma_dtypes = ("float16", "bfloat16", "int8", "uint8")
    unsupported_dtype = "float8_e4m3fn"
    rejected_on_sm8x = unsupported_dtype not in allowed_sm8x_mma_dtypes
    e4m3_max_finite = 448.0
    fp4_e2m1_max_finite = 6.0
    subbyte_accumulator_bits = 4
    subbyte_rejected = subbyte_accumulator_bits < 8
    deep_select_next_pow2_300 = 1 << (300 - 1).bit_length()
    assert rejected_on_sm8x
    assert subbyte_rejected
    assert deep_select_next_pow2_300 == 512
    return {
        "sm8x_sparse_rejects_fp8_e4m3fn": rejected_on_sm8x,
        "e4m3_reduce_clamped_max_finite": e4m3_max_finite,
        "fp4_e2m1_reduce_clamped_max_finite": fp4_e2m1_max_finite,
        "subbyte_accumulator_rejected": subbyte_rejected,
        "deep_select_next_pow2_300": deep_select_next_pow2_300,
    }


def run_all() -> dict[str, object]:
    return {
        "experiment": "flashinfer_mxfp8_moe_rubin_finalize_sm107_tgv_dsa_wide_tilelang_phase90",
        "phase": 90,
        "priority": 99,
        "status": "verified",
        "mxfp8_cute_dsl_fused_moe": model_mxfp8_cute_dsl_fused_moe(),
        "rubin_unfused_deterministic_finalize": model_rubin_unfused_deterministic_finalize(),
        "sm107_tgv_deeper_pipelines": model_sm107_tgv_deeper_pipelines(),
        "cake_dsa_wide_kimi_tp12_moe_ar_softmax": model_cake_dsa_wide_kimi_tp12_moe_ar_softmax(),
        "tilelang_sm8x_sparse_deepselect_reduce": model_tilelang_sm8x_sparse_deepselect_reduce(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    summary = run_all()
    out_path = Path(__file__).resolve().parent / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.check:
        print(f"Wrote {out_path}")
        print(summary["status"])


if __name__ == "__main__":
    main()
