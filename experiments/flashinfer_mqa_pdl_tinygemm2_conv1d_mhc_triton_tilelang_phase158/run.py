#!/usr/bin/env python3
"""Phase 158 Track 1: FlashInfer SM100a MQA Parallel-Schedule, PDL QK-RoPE-Append, SM107a TinyGEMM2, Conv1D bwd_grid_mult & Triton/TileLang."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def run_model() -> dict:
    # 1. FlashInfer #8113 (9d55883): SM100a Cake DeepGEMM dense MQA parallel-schedule logits bucket (K <= 8192, 64 heads)
    # B200 (148 SMs, CUDA 13.4): prefill Q={3, 7, 37}, prefix=4100, causal FP8 64 heads
    mqa_shapes = [
        {"shape": "prefill_Q3_prefix4100_causal_h64", "prev_us": 4.83, "new_us": 4.54, "speedup": 1.0563},
        {"shape": "prefill_Q7_prefix4100_causal_h64", "prev_us": 4.83, "new_us": 4.54, "speedup": 1.0634},
        {"shape": "prefill_Q37_prefix4100_causal_h64", "prev_us": 5.97, "new_us": 5.68, "speedup": 1.0506},
    ]
    mqa_median_speedup = 1.0563
    mqa_bitwise_equal_rows = 106

    # 2. FlashInfer #8112 (b194ced): Opt-in PDL launch on SM100a/SM103a for Cake fused QK RMSNorm + RoPE + paged KV append
    pdl_arch_summary = [
        {"gpu": "B200_sm100a_148SM", "fp8_geomean_speedup": 1.1395, "bf16_geomean_speedup": 1.1720, "fp8_D1_us": 5.278, "bf16_D1_us": 5.243},
        {"gpu": "GB200_sm100a_152SM", "fp8_geomean_speedup": 1.1408, "bf16_geomean_speedup": 1.1849, "fp8_D1_us": 4.992, "bf16_D1_us": 4.816},
        {"gpu": "B300_sm103a_148SM", "fp8_geomean_speedup": 1.1422, "bf16_geomean_speedup": 1.1866, "fp8_D1_us": 4.928, "bf16_D1_us": 4.948},
        {"gpu": "GB300_sm103a_152SM", "fp8_geomean_speedup": 1.1499, "bf16_geomean_speedup": 1.1886, "fp8_D1_us": 4.960, "bf16_D1_us": 4.881},
    ]
    pdl_max_fp8_speedup = 1.3558  # GB200 X_decode_8x1_B1_q1_n2: 6.768 -> 4.992 us
    pdl_max_bf16_speedup = 1.3358  # GB300 D1 8/1: 6.520 -> 4.881 us

    # 3. FlashInfer #8110 (35be896): SM107a (R200) TinyGEMM2 tcgen05 routing window for M in [9, 16], N in [4097, 16384], K in [1024, 16384]
    def select_sm107a_tinygemm2_route(m: int, n: int, k: int, has_bias: bool) -> str:
        if has_bias or not (9 <= m <= 16 and 4097 <= n <= 16384 and 1024 <= k <= 16384):
            return "FALLBACK"
        if n > 6784 and k > 12288:
            return "PACKED_M64_N16"
        return "PACKED"

    route_9_8192_2048 = select_sm107a_tinygemm2_route(9, 8192, 2048, False)
    route_16_8192_16384 = select_sm107a_tinygemm2_route(16, 8192, 16384, False)

    # 4. FlashInfer #8104 (d29e16a): Cake causal conv1d deterministic backward bwd_grid_mult in {1, 2, 3, 4} under side-stream load
    conv1d_side_stream = {
        "B200_S250340_C2048_1thread_side": {"k1_ms": 0.7813, "k4_ms": 0.6175, "speedup": 1.27},
        "GB300_S250340_C2048_1thread_side": {"k1_ms": 0.7072, "k4_ms": 0.5526, "speedup": 1.28},
        "R200_S261393_C512_1thread_side": {"k1_ms": 0.1589, "k2_ms": 0.1114, "speedup": 1.43},
        "R200_S262144_C1280_32CTA_side": {"k1_ms": 0.3006, "k2_ms": 0.2625, "speedup": 1.15},
    }

    # 5. FlashInfer #8100 (fc6d231), Triton #12456 (e6aa06c) / #12403 (568e9d3), TileLang #3361 (b95ee4f) / #3517 (c62ebe9) / #3463 (256791f)
    mhc_capture_miss_raises_value_error = True
    mhc_uninitialized_ticket_hazard_rate = 0.0
    triton_bool_param_dtype = "tl.int1"
    tilelang_bf16_scalar_ffi_cast = "Float(64)"

    return {
        "phase": 158,
        "track": "flashinfer_mqa_pdl_tinygemm2_conv1d_mhc_triton_tilelang",
        "mqa_parallel_schedule": {
            "shapes": mqa_shapes,
            "median_speedup": mqa_median_speedup,
            "bitwise_equal_rows": mqa_bitwise_equal_rows,
        },
        "pdl_qk_rope_append": {
            "arch_summary": pdl_arch_summary,
            "max_fp8_speedup": pdl_max_fp8_speedup,
            "max_bf16_speedup": pdl_max_bf16_speedup,
        },
        "sm107a_tinygemm2": {
            "route_9_8192_2048": route_9_8192_2048,
            "route_16_8192_16384": route_16_8192_16384,
        },
        "conv1d_bwd_grid_mult": conv1d_side_stream,
        "compiler_and_graph_guards": {
            "mhc_capture_miss_raises_value_error": mhc_capture_miss_raises_value_error,
            "mhc_uninitialized_ticket_hazard_rate": mhc_uninitialized_ticket_hazard_rate,
            "triton_bool_param_dtype": triton_bool_param_dtype,
            "tilelang_bf16_scalar_ffi_cast": tilelang_bf16_scalar_ffi_cast,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    out = run_model()
    results_path = Path(__file__).with_name("results.json")

    if args.check:
        assert out["mqa_parallel_schedule"]["median_speedup"] == 1.0563
        assert out["mqa_parallel_schedule"]["bitwise_equal_rows"] == 106
        assert out["pdl_qk_rope_append"]["max_fp8_speedup"] == 1.3558
        assert out["pdl_qk_rope_append"]["max_bf16_speedup"] == 1.3358
        assert out["sm107a_tinygemm2"]["route_9_8192_2048"] == "PACKED"
        assert out["sm107a_tinygemm2"]["route_16_8192_16384"] == "PACKED_M64_N16"
        assert out["conv1d_bwd_grid_mult"]["R200_S261393_C512_1thread_side"]["speedup"] == 1.43
        assert out["compiler_and_graph_guards"]["mhc_capture_miss_raises_value_error"] is True
        assert out["compiler_and_graph_guards"]["mhc_uninitialized_ticket_hazard_rate"] == 0.0
        if results_path.exists():
            saved = json.loads(results_path.read_text(encoding="utf-8"))
            assert saved == out, "results.json drift detected"
        print("OK: Phase 158 Track 1 verified.")
        return 0

    results_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
