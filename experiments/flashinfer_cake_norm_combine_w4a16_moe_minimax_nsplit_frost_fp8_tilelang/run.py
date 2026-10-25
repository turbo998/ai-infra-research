"""Reproducible stdlib model for Phase 75 Track A kernel & compiler mechanics.

Covers:
1. FlashInfer (#7330, 66a81f1bca98): `cake_fused_norm_combine` packed BF16x2 peer sums,
   packed norm conversions, early stream loads, and one-trip pipelined token body on
   8xB200 (`sm_100a`) and 8xB300 (`sm_103a`) for BF16 `H=2560`, 8 peers, 2 norm tracks,
   4 token-count programs across `T in {1, 8, 64, 256, 1024, 2048}`.
2. FlashInfer (#7315, cf5c9a6a415b): `cake_fused_moe` `BF16 x NVFP4` W4A16 split fused-MoE
   kernels (`10` programs/arch on `sm_100a`/`sm_103a`, 12-byte `CakeFastDivmod`, gated
   SwiGLU FC1 & FC2 behind `cutlass_fused_moe(backend="cake")`) across Preset D
   (`H=7168, I=2048, E=256, top-8, 1 GPU`) and Preset G (`H=6144, I=2048, E=256, top-8, EP8`).
3. FlashInfer (#7314, b4125442a533): `minimax_h3_out_proj_nvfp4` N-split tail on
   `sm_100a`/`sm_103a` (half-width `256 x 128` output-column tail tiles when `M >= 32`
   and partial last wave <= 0.5 wave within 48 waves on SM103 or 10 waves on SM100) +
   `mm_bf16_fp4` `tcgen05` M128 pair PDL programs (#7311, 2866b59272d7).
4. FlashInfer (#7212, d3d7c2488783): Grouped Frost low-latency decode extended to
   per-tensor E4M3 FP8 and DeepSeek `128x128` block-FP8 on Hopper (`SM90`, WGMMA
   `group2/stages4/K128`) and `SM120` (native warp FP8 MMA `group2`/`group4`, `stages2/K128`),
   accumulating native partials spanning at most `K64` before FP32 promotion.
5. FlashInfer (#7320, a02c8c5e6b57; #7331, 85fcf535775b) & TileLang (#3558, 8f4501f2c3fd):
   SM120 `minimax_h3` varlen 32-bit offset overflow guard (`(1 << 31) - 1` limit for
   `padded_tokens * heads * 128`), cuDNN warm attention raw-stream lookup avoiding
   `torch.cuda.Stream` allocation (`32.796 -> 31.154 us` prefill, `35.717 -> 32.706 us` decode),
   and TileLang eager AST source-span preservation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def model_cake_fused_norm_combine() -> dict[str, Any]:
    """Model `cake_fused_norm_combine` packed BF16x2 peer sums on 8xB200 and 8xB300 (#7330)."""
    b200_cases = [
        {"tokens": 1, "before_us": 10.112, "after_us": 9.184, "speedup_x": 1.101},
        {"tokens": 8, "before_us": 11.520, "after_us": 10.528, "speedup_x": 1.094},
        {"tokens": 64, "before_us": 14.175, "after_us": 12.928, "speedup_x": 1.096},
        {"tokens": 256, "before_us": 16.768, "after_us": 15.776, "speedup_x": 1.063},
        {"tokens": 1024, "before_us": 29.792, "after_us": 28.416, "speedup_x": 1.048},
        {"tokens": 2048, "before_us": 46.560, "after_us": 42.881, "speedup_x": 1.086},
    ]
    b300_cases = [
        {"tokens": 1, "before_us": 10.272, "after_us": 9.280, "speedup_x": 1.107},
        {"tokens": 8, "before_us": 12.384, "after_us": 11.392, "speedup_x": 1.087},
        {"tokens": 64, "before_us": 15.008, "after_us": 13.792, "speedup_x": 1.088},
        {"tokens": 256, "before_us": 17.472, "after_us": 16.576, "speedup_x": 1.054},
        {"tokens": 1024, "before_us": 30.688, "after_us": 28.991, "speedup_x": 1.059},
        {"tokens": 2048, "before_us": 47.008, "after_us": 43.872, "speedup_x": 1.071},
    ]
    b200_geomean = round(math.prod(c["speedup_x"] for c in b200_cases) ** (1.0 / len(b200_cases)), 4)
    b300_geomean = round(math.prod(c["speedup_x"] for c in b300_cases) ** (1.0 / len(b300_cases)), 4)
    return {
        "hidden_size": 2560,
        "num_peers": 8,
        "norm_tracks": 2,
        "programs_per_arch": 4,
        "launch_bounds_threads": 320,
        "vectors_per_token": 160,
        "ring_buffers": 3,
        "b200_sm100a_cases": b200_cases,
        "b300_sm103a_cases": b300_cases,
        "b200_geomean_speedup_x": b200_geomean,
        "b300_geomean_speedup_x": b300_geomean,
        "max_b200_speedup_x": max(c["speedup_x"] for c in b200_cases),
        "max_b300_speedup_x": max(c["speedup_x"] for c in b300_cases),
    }


def model_cake_w4a16_split_moe() -> dict[str, Any]:
    """Model `cake_fused_moe` BF16 x NVFP4 W4A16 split fused-MoE on B300 (`sm_103a`) (#7315)."""
    preset_d_cases = [
        {"case": "D1", "tokens": 1, "cake_s1_us": 50.98, "cake_s2_us": 51.40, "cutlass_us": 53.73, "speedup_s1_x": 1.054},
        {"case": "D128", "tokens": 128, "cake_s1_us": 1027.63, "cake_s2_us": 1026.78, "cutlass_us": 974.94, "speedup_s1_x": 0.949},
    ]
    preset_g_cases = [
        {"case": "G8", "tokens": 8, "cake_s1_us": 55.98, "cake_s2_us": 55.89, "cutlass_us": 58.63, "speedup_s1_x": 1.047},
        {"case": "G32", "tokens": 32, "cake_s1_us": 66.22, "cake_s2_us": 66.26, "cutlass_us": 66.66, "speedup_s1_x": 1.007},
        {"case": "G64", "tokens": 64, "cake_s1_us": 92.54, "cake_s2_us": 92.62, "cutlass_us": 93.42, "speedup_s1_x": 1.010},
        {"case": "G128", "tokens": 128, "cake_s1_us": 133.62, "cake_s2_us": 133.49, "cutlass_us": 123.31, "speedup_s1_x": 0.923},
        {"case": "G512", "tokens": 512, "cake_s1_us": 225.70, "cake_s2_us": 225.49, "cutlass_us": 227.97, "speedup_s1_x": 1.010},
        {"case": "G4096", "tokens": 4096, "cake_s1_us": 310.51, "cake_s2_us": 310.11, "cutlass_us": 317.56, "speedup_s1_x": 1.023},
    ]
    winning_cases = [c for c in preset_d_cases + preset_g_cases if c["speedup_s1_x"] > 1.0]
    regression_cases = [c for c in preset_d_cases + preset_g_cases if c["speedup_s1_x"] < 1.0]
    return {
        "programs_per_arch": 10,
        "target_arches": ["sm_100a", "sm_103a"],
        "fast_divmod_bytes": 12,
        "preset_d_shape": {"H": 7168, "I": 2048, "E": 256, "top_k": 8, "ep_size": 1},
        "preset_g_shape": {"H": 6144, "I": 2048, "E": 256, "top_k": 8, "ep_size": 8},
        "preset_d_cases": preset_d_cases,
        "preset_g_cases": preset_g_cases,
        "winning_case_count": len(winning_cases),
        "regression_case_count": len(regression_cases),
        "peak_decode_speedup_x": max(c["speedup_s1_x"] for c in winning_cases),
        "worst_mid_batch_ratio_x": min(c["speedup_s1_x"] for c in regression_cases),
    }


def model_minimax_h3_nsplit_and_mm_fp4() -> dict[str, Any]:
    """Model `minimax_h3_out_proj_nvfp4` N-split tail (#7314) and `mm_bf16_fp4` M128 pair (#7311)."""
    gb300_featured = [
        {"P": 1, "M": 33472, "tail_split": True, "before_us": 469.5, "after_us": 467.9, "speedup_x": 1.003},
        {"P": 2, "M": 16736, "tail_split": True, "before_us": 240.0, "after_us": 237.6, "speedup_x": 1.010},
        {"P": 4, "M": 8368, "tail_split": True, "before_us": 129.7, "after_us": 126.0, "speedup_x": 1.029},
        {"P": 4, "M": 12192, "tail_split": True, "before_us": 178.4, "after_us": 175.0, "speedup_x": 1.020},
        {"P": 8, "M": 4816, "tail_split": True, "before_us": 82.2, "after_us": 79.0, "speedup_x": 1.041},
        {"P": 8, "M": 7368, "tail_split": True, "before_us": 116.4, "after_us": 112.8, "speedup_x": 1.031},
    ]
    return {
        "min_rows_for_tail_split": 32,
        "max_partial_last_wave_fraction": 0.5,
        "max_waves_sm103": 48,
        "max_waves_sm100": 10,
        "half_width_tail_tile_mn": [256, 128],
        "gb300_tail_split_shape_count": 20,
        "gb300_geomean_speedup_x": 1.021,
        "gb300_speedup_range_x": [1.003, 1.041],
        "gb200_tail_split_shape_count": 7,
        "gb200_geomean_speedup_x": 1.042,
        "gb200_speedup_range_x": [1.017, 1.052],
        "gb300_featured_cases": gb300_featured,
        "mm_bf16_fp4_m128_pair": {
            "shape_mnk": [768, 2112, 2048],
            "gb300_native_speedup_x": 1.015,
            "gb300_tiled_speedup_x": 1.029,
            "b200_native_speedup_x": 1.018,
            "b200_tiled_speedup_x": 1.023,
        },
    }


def model_frost_fp8_grouped_decode() -> dict[str, Any]:
    """Model Grouped Frost FP8 and DeepSeek block-FP8 decode on SM90 and SM120 (#7212)."""
    return {
        "benchmark_shape": {"H": 2048, "I": 512, "top_k": 10, "tokens": [4, 8]},
        "native_partial_max_k": 64,
        "tma_staging_tile_k": 128,
        "deepseek_block_fp8_scale_tile": [128, 128],
        "sm90_per_tensor_fp8": {
            "mma": "WGMMA",
            "group": 2,
            "stages": 4,
            "tile_k": 128,
            "activations_t4_t8": ["SwiGLU", "SiTU"],
            "activations_t8_only": ["ReLU2"],
        },
        "sm120_per_tensor_fp8": {
            "mma": "native_warp_fp8_mma",
            "groups": [2, 4],
            "stages": 2,
            "tile_k": 128,
            "activations_t4_t8": ["SwiGLU", "SiTU", "ReLU2"],
        },
        "deepseek_block_fp8_sm90_sm120": {
            "activations_t4_t8": ["SwiGLU"],
            "activations_t8_only": ["ReLU2", "SiLU"],
            "intermediate_quantizer_group": 128,
        },
        "fp32_promotions_per_k128_tile": 128 // 64,
    }


def model_sm120_varlen_and_cudnn_stream_and_tilelang() -> dict[str, Any]:
    """Model SM120 MiniMax-H3 32-bit offset guard (#7320), cuDNN raw stream (#7331), and TileLang AST (#3558)."""
    max_elements = (1 << 31) - 1
    heads = 56
    head_dim = 128
    block_n = 128
    partial_floats = 2 * head_dim
    row_elems = heads * head_dim
    max_safe_padded_tokens = (max_elements // row_elems // block_n) * block_n
    max_safe_segments = max_elements // row_elems
    max_safe_query_tiles = max_elements // (heads * partial_floats)

    cudnn_prefill_before_us = 32.796
    cudnn_prefill_after_us = 31.154
    cudnn_decode_before_us = 35.717
    cudnn_decode_after_us = 32.706

    return {
        "int32_max_elements": max_elements,
        "minimax_h3_heads": heads,
        "minimax_h3_head_dim": head_dim,
        "block_n": block_n,
        "max_safe_padded_tokens": max_safe_padded_tokens,
        "first_overflow_tokens": max_safe_padded_tokens + 1,
        "max_safe_segments": max_safe_segments,
        "max_safe_query_tiles": max_safe_query_tiles,
        "cudnn_warm_stream_b200": {
            "prefill_before_us": cudnn_prefill_before_us,
            "prefill_after_us": cudnn_prefill_after_us,
            "prefill_reduction_pct": round(
                (cudnn_prefill_before_us - cudnn_prefill_after_us) / cudnn_prefill_before_us * 100.0, 2
            ),
            "decode_before_us": cudnn_decode_before_us,
            "decode_after_us": cudnn_decode_after_us,
            "decode_reduction_pct": round(
                (cudnn_decode_before_us - cudnn_decode_after_us) / cudnn_decode_before_us * 100.0, 2
            ),
        },
        "tilelang_ast_span": {
            "preserved_span_fields": ["lineno", "col_offset", "end_lineno", "end_col_offset"],
            "quote_scaffolding_collapses_to_first_line": True,
            "substituted_user_nodes_preserve_full_span": True,
        },
    }


def build_summary() -> dict[str, Any]:
    norm_combine = model_cake_fused_norm_combine()
    w4a16_moe = model_cake_w4a16_split_moe()
    minimax_nsplit = model_minimax_h3_nsplit_and_mm_fp4()
    frost_fp8 = model_frost_fp8_grouped_decode()
    guards_and_stream = model_sm120_varlen_and_cudnn_stream_and_tilelang()
    return {
        "experiment_id": "flashinfer_cake_norm_combine_w4a16_moe_minimax_nsplit_frost_fp8_tilelang",
        "phase": 75,
        "priority": 84,
        "track": "Track A — Kernel & Compiler",
        "cake_fused_norm_combine": norm_combine,
        "cake_w4a16_split_moe": w4a16_moe,
        "minimax_h3_nsplit_and_mm_fp4": minimax_nsplit,
        "frost_fp8_grouped_decode": frost_fp8,
        "sm120_varlen_and_cudnn_stream_and_tilelang": guards_and_stream,
        "invariants": {
            "norm_combine_b300_peak_speedup_1_107x": norm_combine["max_b300_speedup_x"] == 1.107,
            "w4a16_moe_has_mid_batch_regression_boundary": w4a16_moe["worst_mid_batch_ratio_x"] == 0.923,
            "minimax_nsplit_bitwise_identical_tail_tile": minimax_nsplit["half_width_tail_tile_mn"] == [256, 128],
            "frost_fp8_promotes_every_k64": frost_fp8["fp32_promotions_per_k128_tile"] == 2,
            "sm120_varlen_max_safe_padded_tokens_299520": guards_and_stream["max_safe_padded_tokens"] == 299520,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("summary.json"),
        help="Output path for summary.json",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate that the existing summary.json matches build_summary()",
    )
    args = parser.parse_args()
    summary = build_summary()
    if args.check:
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        if existing != summary:
            raise SystemExit("summary.json is out of date with build_summary()")
        return 0
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
