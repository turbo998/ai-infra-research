"""Phase 120 Track A stdlib experiment: FlashInfer GB300 SF-Ring FP4 GEMM,
CAKE FMHA Paged Decode, Rubin MLA NVFP4 Decode, SVDQuant One-Wave Tail,
MiniMax H3 Varlen, Warp Decode, B12X MoE Bucketing, cuDNN 1.32 QSA & TileLang ROCm.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def model_gb300_sf_ring_fp4_gemm() -> dict[str, Any]:
    """Model FlashInfer 4ca4e6e: GB300 sm_103a cake_mm_fp4 with separate 5-stage SF ring."""
    gemm_rows = [
        {"shape": "16384x7168x2048", "before_us": 62.33, "after_us": 56.35},
        {"shape": "16384x4096x7168", "before_us": 115.65, "after_us": 107.61},
        {"shape": "16384x2048x7168", "before_us": 62.74, "after_us": 60.62},
    ]
    for r in gemm_rows:
        r["speedup"] = round(r["before_us"] / r["after_us"], 4)

    chain_rows = [
        {"shape": "28672x8192x2048", "before_us": 156.53, "after_us": 144.46},
        {"shape": "16384x7168x2048", "before_us": 81.33, "after_us": 75.59},
        {"shape": "16384x4096x7168", "before_us": 135.42, "after_us": 127.64},
    ]
    for r in chain_rows:
        r["speedup"] = round(r["before_us"] / r["after_us"], 4)

    return {
        "sf_ring_arch": "sm_103a",
        "gb300_sm_count": 152,
        "num_tma_pipe_stages": 6,
        "num_sf_pipe_stages": 5,
        "epilogue_mode": "register_direct_gemm_2cta_ptr",
        "gemm_rows": gemm_rows,
        "max_gemm_speedup": max(r["speedup"] for r in gemm_rows),
        "quant_gemm_chain_rows": chain_rows,
        "max_chain_speedup": max(r["speedup"] for r in chain_rows),
    }


def model_cake_fmha_paged_decode() -> dict[str, Any]:
    """Model FlashInfer 5e0e99b: 27 paged decode attention programs on sm_100a & sm_103a."""
    bf16_b4_kv1024_before_us = 13.44
    bf16_b4_kv1024_after_us = 7.68
    dev_scale_before_us = 59999.70
    dev_scale_after_us = 169.89
    fp8_b256_kv8192_cake_us = 317.31
    fp8_b256_kv8192_trtllm_us = 321.45
    return {
        "total_aot_programs": 27,
        "programs_per_target": 19,
        "targets": ["sm_100a", "sm_103a"],
        "bf16_b4_kv1024_before_us": bf16_b4_kv1024_before_us,
        "bf16_b4_kv1024_after_us": bf16_b4_kv1024_after_us,
        "bf16_b4_kv1024_speedup": round(bf16_b4_kv1024_before_us / bf16_b4_kv1024_after_us, 4),
        "device_scale_before_us": dev_scale_before_us,
        "device_scale_after_us": dev_scale_after_us,
        "device_scale_speedup": round(dev_scale_before_us / dev_scale_after_us, 2),
        "fp8_b256_kv8192_cake_us": fp8_b256_kv8192_cake_us,
        "fp8_b256_kv8192_trtllm_us": fp8_b256_kv8192_trtllm_us,
        "fp8_vs_trtllm_speedup": round(fp8_b256_kv8192_trtllm_us / fp8_b256_kv8192_cake_us, 4),
    }


def model_rubin_mla_nvfp4_decode_and_svdquant() -> dict[str, Any]:
    """Model FlashInfer f6587bc (Rubin MLA NVFP4 decode) & 03ebbc6 (SVDQuant one-wave tail)."""
    # Rubin sm_107a 208-SM board: units = 50, pairs = 4
    units = 50
    pairs = 4
    old_pdl_limit = units
    new_pdl_limit = units + pairs

    # SVDQuant (4096, 3072, 12288, rank=32) on 152 SMs (76 CTA pairs)
    m, n, k, rank = 4096, 3072, 12288, 32
    tiles_256 = (m // 256) * (n // 256)  # 16 * 12 = 192
    cta_pairs = 76
    full_wave_tiles = 2 * cta_pairs  # 152
    tail_tiles = tiles_256 - full_wave_tiles  # 40
    tail_cols = tail_tiles * 256  # 10240
    pieces_192 = 8
    pieces_128 = 68
    repacked_cols = pieces_192 * 192 + pieces_128 * 128
    repacked_units = pieces_192 + pieces_128

    svd_before_us = 62.98
    svd_after_us = 59.17

    return {
        "rubin_arch": "sm_107a",
        "rubin_old_pdl_max_items": old_pdl_limit,
        "rubin_new_pdl_max_items": new_pdl_limit,
        "rubin_column_merge_bytes_per_split": 8,
        "rubin_h192_q1_b2_kv128k_speedup": 1.067,
        "svdquant_shape": [m, n, k, rank],
        "svdquant_total_256_tiles": tiles_256,
        "svdquant_cta_pairs": cta_pairs,
        "svdquant_tail_tiles": tail_tiles,
        "svdquant_tail_cols": tail_cols,
        "svdquant_repacked_cols": repacked_cols,
        "svdquant_repacked_units": repacked_units,
        "svdquant_single_wave_fit": repacked_cols == tail_cols and repacked_units == cta_pairs,
        "svdquant_gb200_before_us": svd_before_us,
        "svdquant_gb200_after_us": svd_after_us,
        "svdquant_gb200_speedup": round(svd_before_us / svd_after_us, 4),
    }


def model_minimax_h3_and_warp_decode() -> dict[str, Any]:
    """Model FlashInfer d949946 (MiniMax H3 varlen) & 7de971a (cake_warp_decode)."""
    warp_before_us = 16.08
    warp_after_us = 15.12
    return {
        "minimax_nvfp4_geomean_speedup_range": [1.0589, 1.0646],
        "minimax_tiny_bf16_gb300_speedup_range": [2.17, 2.25],
        "minimax_tiny_bf16_b200_speedup_range": [2.24, 2.34],
        "minimax_ticket_mailbox_words": [6, 7],
        "warp_decode_route_pack_rows_sm100": 129,
        "warp_decode_route_pack_rows_sm103": 10,
        "warp_decode_early_load_rows_sm100": 46,
        "warp_decode_early_load_rows_sm103": 12,
        "warp_decode_e512_i512_k10_before_us": warp_before_us,
        "warp_decode_e512_i512_k10_after_us": warp_after_us,
        "warp_decode_e512_i512_k10_speedup": round(warp_before_us / warp_after_us, 4),
    }


def bucket_workspace_rows(routed_rows: int, max_safe_rows: int = 131072) -> int:
    """Replicate FlashInfer 959a113 _bucket_workspace_rows power-of-two bucketing."""
    bucketed = 1 << (max(128, int(routed_rows)) - 1).bit_length()
    if bucketed > max_safe_rows:
        return int(routed_rows)
    return bucketed


def model_b12x_moe_bucketing_cudnn_qsa_and_tilelang_rocm() -> dict[str, Any]:
    """Model FlashInfer 959a113 (b12x MoE bucket), c2ba71f (cuDNN 1.32 QSA),
    and TileLang 040b305 / 015caa2 (ROCm current_device & pip-SDK ROCm 10).
    """
    sample_routed_rows = list(range(64, 4097, 17))
    unbucketed_compiles = len(set(sample_routed_rows))
    bucketed_shapes = sorted({bucket_workspace_rows(r) for r in sample_routed_rows})
    bucketed_compiles = len(bucketed_shapes)

    return {
        "b12x_sample_sweep_count": unbucketed_compiles,
        "b12x_bucketed_compile_count": bucketed_compiles,
        "b12x_bucketed_shapes": bucketed_shapes,
        "b12x_compile_reduction_ratio": round(unbucketed_compiles / bucketed_compiles, 2),
        "b12x_fallback_above_limit": bucket_workspace_rows(90000, max_safe_rows=65536),
        "cudnn_qsa_min_version": "1.32.0",
        "cudnn_qsa_bitmap_tokens_per_uint8": 8,
        "cudnn_qsa_block_size": 4,
        "cudnn_qsa_tokens_per_uint8_byte": 32,
        "cudnn_qsa_kernel_boundary_tokens": [512, 513],
        "tilelang_rocm_uses_current_device": True,
        "tilelang_rocm10_resolution_tiers": [
            "explicit_rocm_path",
            "find_rocm_path_with_10s_timeout",
            "torch_device_properties_gcnArchName",
            "fallback_gfx900",
        ],
    }


def run_all() -> dict[str, Any]:
    return {
        "phase": 120,
        "track": "track_a",
        "title": "FlashInfer GB300 SF-Ring FP4, CAKE FMHA Paged Decode, Rubin MLA, SVDQuant Tail, B12X MoE, cuDNN 1.32 QSA & TileLang ROCm 10",
        "gb300_sf_ring_fp4_gemm": model_gb300_sf_ring_fp4_gemm(),
        "cake_fmha_paged_decode": model_cake_fmha_paged_decode(),
        "rubin_mla_and_svdquant": model_rubin_mla_nvfp4_decode_and_svdquant(),
        "minimax_h3_and_warp_decode": model_minimax_h3_and_warp_decode(),
        "b12x_qsa_tilelang_rocm": model_b12x_moe_bucketing_cudnn_qsa_and_tilelang_rocm(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify deterministic invariants")
    args = parser.parse_args()

    summary = run_all()
    out_path = Path(__file__).resolve().parent / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    if args.check:
        assert summary["gb300_sf_ring_fp4_gemm"]["max_gemm_speedup"] == 1.1061
        assert summary["cake_fmha_paged_decode"]["bf16_b4_kv1024_speedup"] == 1.75
        assert summary["rubin_mla_and_svdquant"]["svdquant_single_wave_fit"] is True
        assert summary["b12x_qsa_tilelang_rocm"]["b12x_bucketed_compile_count"] == 6

    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
