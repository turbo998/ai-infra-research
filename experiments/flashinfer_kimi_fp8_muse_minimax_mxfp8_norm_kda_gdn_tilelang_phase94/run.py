#!/usr/bin/env python3
"""Phase 94 Track 1 deterministic stdlib experiment:
FlashInfer Kimi-K3 FP8 projection registry folding (#7399), Muse Spark SM107a PDL norm (#7400),
MiniMax-H3 SM103 256-bit residual out-proj (#7397) & SM120 FP8/NVFP4 pre-attn (#7347),
grouped MXFP8 224-thread quantize (#7391), fused norm-combine min.s16x2 poll (#7390),
BF16-input FP32-state KDA prefill (#7388), SM8x 4-stage chunk-parallel GDN (#5109),
and TileLang Ascend exact buffer placement (#3607) & Int64Promoter region bounds (#3625).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def model_kimi_fp8_muse_minimax() -> dict[str, object]:
    # FlashInfer #7399: cake_kimi_k3_fp8_projection launcher & twin kernel folding
    launchers_before = 54
    launchers_after = 5
    launcher_reduction_pct = round((1.0 - launchers_after / launchers_before) * 100.0, 2)
    kernels_before = 75
    kernels_after = 70
    decode_m_buckets_before = [1, 8, 64, 256]
    decode_m_buckets_after = [1, 8, 64, 256, 512, 1024, 2048]

    # FlashInfer #7400: cake_muse_spark_norm on sm_107a (Rubin R200, 212 SMs)
    sm107a_sms = 212
    plain_band_thresholds = {"h128_min_tokens": 384, "h32_min_tokens": 768, "h16_min_tokens": 2048}

    # FlashInfer #7397: MiniMax-H3 SM103 256-bit residual loads (76 tile-pair waves on 152 SMs)
    sm103_sms = 152
    tile_pairs_per_wave = sm103_sms // 2  # 76
    nvfp4_tailsplit_min_waves = 8
    nvfp4_tailsplit_min_pairs = nvfp4_tailsplit_min_waves * tile_pairs_per_wave  # 608
    nvfp4_wholetile_max_waves = 48
    nvfp4_wholetile_max_pairs = nvfp4_wholetile_max_waves * tile_pairs_per_wave  # 3648
    residual_loads_per_16col_before = 2  # two 128-bit loads
    residual_loads_per_16col_after = 1   # one 256-bit load

    # FlashInfer #7347: MiniMax-H3 SM120 FP8/NVFP4 pre-attention (RTX 5090)
    fp8_bf16_m33472_before_ms = 18.6993
    fp8_bf16_m33472_after_ms = 13.6945
    fp8_bf16_speedup_x = round(fp8_bf16_m33472_before_ms / fp8_bf16_m33472_after_ms, 4)
    nvfp4_bf16_m33472_before_ms = 7.9134
    nvfp4_bf16_m33472_after_ms = 7.2648
    nvfp4_bf16_speedup_x = round(nvfp4_bf16_m33472_before_ms / nvfp4_bf16_m33472_after_ms, 4)

    assert launcher_reduction_pct == 90.74
    assert nvfp4_tailsplit_min_pairs == 608 and nvfp4_wholetile_max_pairs == 3648
    assert fp8_bf16_speedup_x == 1.3655 and nvfp4_bf16_speedup_x == 1.0893

    return {
        "launchers_before": launchers_before,
        "launchers_after": launchers_after,
        "launcher_reduction_pct": launcher_reduction_pct,
        "kernels_before": kernels_before,
        "kernels_after": kernels_after,
        "decode_m_buckets_before": decode_m_buckets_before,
        "decode_m_buckets_after": decode_m_buckets_after,
        "sm107a_sms": sm107a_sms,
        "plain_band_thresholds": plain_band_thresholds,
        "nvfp4_tailsplit_min_pairs": nvfp4_tailsplit_min_pairs,
        "nvfp4_wholetile_max_pairs": nvfp4_wholetile_max_pairs,
        "residual_loads_per_16col_before": residual_loads_per_16col_before,
        "residual_loads_per_16col_after": residual_loads_per_16col_after,
        "fp8_bf16_m33472_before_ms": fp8_bf16_m33472_before_ms,
        "fp8_bf16_m33472_after_ms": fp8_bf16_m33472_after_ms,
        "fp8_bf16_speedup_x": fp8_bf16_speedup_x,
        "nvfp4_bf16_speedup_x": nvfp4_bf16_speedup_x,
    }


def model_mxfp8_quant_norm_combine_kda_gdn() -> dict[str, object]:
    # FlashInfer #7391: cake_grouped_mxfp8_quantize 224-thread row-tiled CTA for K=7168, B*M >= 16384
    b, m, k = 32, 1024, 7168
    block_scale_elems = 32
    blocks_per_row = k // block_scale_elems  # 224
    total_rows = b * m  # 32768
    ctas_128thread = total_rows * math.ceil(blocks_per_row / 128)  # 65536
    ctas_224thread = total_rows * math.ceil(blocks_per_row / 224)  # 32768
    cta_reduction_x = ctas_128thread // ctas_224thread
    sparse_speedup_max_x = 1.60

    # FlashInfer #7390: cake_fused_norm_combine packed min.s16x2 poll & record-first 4-CTA load
    b200_t1_before_us = 8.128
    b200_t1_after_us = 7.712
    b200_t1_speedup_x = round(b200_t1_before_us / b200_t1_after_us, 3)
    b200_t8_before_us = 10.560
    b200_t8_after_us = 10.048
    b200_t8_speedup_x = round(b200_t8_before_us / b200_t8_after_us, 3)
    stream_program_min_tokens_before = 40
    stream_program_min_tokens_after = 38

    # FlashInfer #7388: BF16-input FP32-state KDA prefill
    gb300_kda_8192_before_us = 275.78
    gb300_kda_8192_after_us = 238.53
    gb300_kda_8192_speedup_x = round(gb300_kda_8192_before_us / gb300_kda_8192_after_us, 2)
    b200_kda_bs4_t64_before_us = 19.68
    b200_kda_bs4_t64_after_us = 16.10
    b200_kda_bs4_t64_speedup_x = round(b200_kda_bs4_t64_before_us / b200_kda_bs4_t64_after_us, 2)

    # FlashInfer #5109: SM8x 4-stage chunk-parallel GDN prefill
    seq_len = 8192
    fused_tile_tokens = 64
    cp_chunk_tokens = 2048
    serial_steps_fused = seq_len // fused_tile_tokens  # 128
    serial_steps_cp_fixup = seq_len // cp_chunk_tokens  # 4
    serial_chain_reduction_x = serial_steps_fused // serial_steps_cp_fixup  # 32

    assert blocks_per_row == 224 and cta_reduction_x == 2
    assert b200_t1_speedup_x == 1.054 and b200_t8_speedup_x == 1.051
    assert gb300_kda_8192_speedup_x == 1.16 and b200_kda_bs4_t64_speedup_x == 1.22
    assert serial_chain_reduction_x == 32

    return {
        "blocks_per_row_k7168": blocks_per_row,
        "ctas_128thread": ctas_128thread,
        "ctas_224thread": ctas_224thread,
        "cta_reduction_x": cta_reduction_x,
        "sparse_speedup_max_x": sparse_speedup_max_x,
        "b200_t1_speedup_x": b200_t1_speedup_x,
        "b200_t8_speedup_x": b200_t8_speedup_x,
        "stream_program_min_tokens_before": stream_program_min_tokens_before,
        "stream_program_min_tokens_after": stream_program_min_tokens_after,
        "gb300_kda_8192_speedup_x": gb300_kda_8192_speedup_x,
        "b200_kda_bs4_t64_speedup_x": b200_kda_bs4_t64_speedup_x,
        "serial_steps_fused": serial_steps_fused,
        "serial_steps_cp_fixup": serial_steps_cp_fixup,
        "serial_chain_reduction_x": serial_chain_reduction_x,
    }


def model_tilelang_ascend_placement_and_int64() -> dict[str, object]:
    # TileLang #3607: Exact on-chip buffer capacities and alignments on Ascend
    scopes = {
        "shared.dyn": {"capacity_bytes": 248 * 1024, "simt_reserved_bytes": 32 * 1024, "alignment": 32},
        "shared.l1": {"capacity_bytes": 512 * 1024, "alignment": 512},
        "shared.l0a": {"capacity_bytes": 64 * 1024, "alignment": 512},
        "shared.l0b": {"capacity_bytes": 64 * 1024, "alignment": 512},
        "shared.l0c": {"capacity_bytes": 256 * 1024, "alignment": 1024},
        "shared.ss": {"capacity_bytes": 3 * 1024, "alignment": 32},
    }
    ub_with_simt_bytes = scopes["shared.dyn"]["capacity_bytes"] - scopes["shared.dyn"]["simt_reserved_bytes"]

    # TileLang #3625: Int64Promoter prevents int32 overflow on half-open endpoint [limit, limit + 1)
    int32_max = (1 << 31) - 1  # 2147483647
    int32_wrapped_endpoint = ((int32_max + 1 + (1 << 31)) % (1 << 32)) - (1 << 31)  # -2147483648
    int64_exact_endpoint = int32_max + 1  # 2147483648
    int32_overflows = int32_wrapped_endpoint < int32_max
    int64_preserves_order = int64_exact_endpoint > int32_max

    assert ub_with_simt_bytes == 216 * 1024
    assert int32_overflows is True and int64_preserves_order is True

    return {
        "scopes": scopes,
        "ub_with_simt_bytes": ub_with_simt_bytes,
        "int32_max": int32_max,
        "int32_wrapped_endpoint": int32_wrapped_endpoint,
        "int64_exact_endpoint": int64_exact_endpoint,
        "int32_overflows": int32_overflows,
        "int64_preserves_order": int64_preserves_order,
    }


def generate_summary() -> dict[str, object]:
    return {
        "experiment": "flashinfer_kimi_fp8_muse_minimax_mxfp8_norm_kda_gdn_tilelang_phase94",
        "phase": 94,
        "priority": 103,
        "provenance": "own-offline",
        "kimi_fp8_muse_minimax": model_kimi_fp8_muse_minimax(),
        "mxfp8_quant_norm_combine_kda_gdn": model_mxfp8_quant_norm_combine_kda_gdn(),
        "tilelang_ascend_placement_and_int64": model_tilelang_ascend_placement_and_int64(),
        "all_invariants_passed": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    summary = generate_summary()
    out_path = Path(__file__).resolve().parent / "summary.json"
    if args.check:
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        assert existing == summary, "Mismatch in summary.json"
        print("OK: flashinfer_kimi_fp8_muse_minimax_mxfp8_norm_kda_gdn_tilelang_phase94 verified")
        return 0
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
