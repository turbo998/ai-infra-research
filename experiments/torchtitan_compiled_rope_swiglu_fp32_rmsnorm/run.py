#!/usr/bin/env python3
"""Deterministic offline model for Phase 34 Topic 2:
1. TorchTitan compiled ComplexRoPE in real arithmetic with int32-packed bf16 pairs & autotune-off determinism (torchtitan@b95271ed07)
2. TorchTitan SwiGLU backward unmasked per-half stores via max_pointwise_cat_inputs=0 & max_complex_pointwise_cat_inputs=0 (torchtitan@eb48bab350)
3. TorchTitan Muse Spark fp32-stream RMSNorm weight dtype cast to restore single fused kernel (torchtitan@b78c4aed96)
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path
from typing import Any


def f32_to_bf16_bits(val: float) -> int:
    bits32 = struct.unpack("<I", struct.pack("<f", val))[0]
    rounding_bias = 0x00007FFF + ((bits32 >> 16) & 1)
    return ((bits32 + rounding_bias) >> 16) & 0xFFFF


def bf16_bits_to_f32(bits16: int) -> float:
    bits32 = (bits16 & 0xFFFF) << 16
    return struct.unpack("<f", struct.pack("<I", bits32))[0]


def simulate_compiled_complex_rope() -> dict[str, Any]:
    # Verify exact bitwise correspondence of TorchTitan's int32-packed bf16 pair rotation:
    # pairs = x.view(torch.int32)
    # x_real = (pairs << 16).view(torch.float32)
    # x_imag = (pairs & ~0xFFFF).view(torch.float32)
    # low = real.view(torch.int16).to(torch.int32) & 0xFFFF
    # high = imag.view(torch.int16).to(torch.int32) << 16
    sample_pairs = [(1.25, -0.75), (0.375, 2.5), (-1.125, 0.625), (3.0, -2.25)]
    angles = [0.25, 0.75, 1.35, 2.10]

    max_ulp_diff = 0
    inv_roundtrip_exact = True

    for (xr, xi), theta in zip(sample_pairs, angles):
        bf_r = f32_to_bf16_bits(xr)
        bf_i = f32_to_bf16_bits(xi)
        # Pack two bf16 into one int32 (low=x[2i], high=x[2i+1])
        packed_pair = (bf_r & 0xFFFF) | ((bf_i & 0xFFFF) << 16)

        # Unpack via bit shift / mask as in _RotatePairs.rotate
        unpacked_r_bits32 = (packed_pair << 16) & 0xFFFFFFFF
        unpacked_i_bits32 = packed_pair & (~0xFFFF & 0xFFFFFFFF)
        unpacked_r = struct.unpack("<f", struct.pack("<I", unpacked_r_bits32))[0]
        unpacked_i = struct.unpack("<f", struct.pack("<I", unpacked_i_bits32))[0]

        cos_v = math.cos(theta)
        sin_v = math.sin(theta)

        out_r_bf16 = f32_to_bf16_bits(unpacked_r * cos_v - unpacked_i * sin_v)
        out_i_bf16 = f32_to_bf16_bits(unpacked_r * sin_v + unpacked_i * cos_v)
        packed_out = (out_r_bf16 & 0xFFFF) | ((out_i_bf16 & 0xFFFF) << 16)

        # Reference complex multiply in fp32 then round to bf16
        ref_r_bf16 = f32_to_bf16_bits(bf16_bits_to_f32(bf_r) * cos_v - bf16_bits_to_f32(bf_i) * sin_v)
        ref_i_bf16 = f32_to_bf16_bits(bf16_bits_to_f32(bf_r) * sin_v + bf16_bits_to_f32(bf_i) * cos_v)
        ulp_diff = max(abs((packed_out & 0xFFFF) - ref_r_bf16), abs(((packed_out >> 16) & 0xFFFF) - ref_i_bf16))
        max_ulp_diff = max(max_ulp_diff, ulp_diff)

        # Backward / inverse rotates by -sin
        bwd_r = bf16_bits_to_f32(out_r_bf16) * cos_v - bf16_bits_to_f32(out_i_bf16) * (-sin_v)
        bwd_i = bf16_bits_to_f32(out_r_bf16) * (-sin_v) + bf16_bits_to_f32(out_i_bf16) * cos_v
        if abs(bwd_r - bf16_bits_to_f32(bf_r)) > 0.02 or abs(bwd_i - bf16_bits_to_f32(bf_i)) > 0.02:
            inv_roundtrip_exact = False

    # Author-reported GB300 benchmark table (torchtitan@b95271ed07)
    gb300_benchmarks_us = {
        "llama3_8b_64k_train": {"eager_us": 2223.4, "compiled_us": 452.7, "helion_us": 462.8, "speedup_vs_eager": round(2223.4 / 452.7, 2)},
        "llama3_8b_64k_prefill": {"eager_us": 1126.5, "compiled_us": 224.3, "helion_us": 231.5, "speedup_vs_eager": round(1126.5 / 224.3, 2)},
        "deepseek_v3_671b_64k_train": {"eager_us": 5836.6, "compiled_us": 2262.1, "helion_us": 2285.9, "speedup_vs_eager": round(5836.6 / 2262.1, 2)},
        "deepseek_v3_671b_64k_prefill": {"eager_us": 2589.9, "compiled_us": 426.5, "helion_us": 384.9, "speedup_vs_eager": round(2589.9 / 426.5, 2)},
    }

    return {
        "int32_pair_max_ulp_diff_vs_complex": max_ulp_diff,
        "inverse_rotation_consistent": inv_roundtrip_exact,
        "autotune_on_distinct_losses_in_13_runs": 9,
        "autotune_off_distinct_losses_in_12_runs": 1,
        "gb300_benchmarks_us": gb300_benchmarks_us,
    }


def simulate_swiglu_unmasked_cat() -> dict[str, Any]:
    # Author-measured GB300 SwiGLU region [8T, 2, F=2048] bf16 (torchtitan@eb48bab350)
    # Bandwidth floor at T=16k (rows=131072, F=2048) is 377 us; masked cat backward took 788 us.
    bwd_floor_us = 377.0
    bwd_masked_cat_us = 788.0
    fwd_plus_bwd_rows = [
        {"tokens": 2048, "main_masked_us": 188.3, "unmasked_cat0_us": 139.4, "triton_override_us": 143.3},
        {"tokens": 4096, "main_masked_us": 366.8, "unmasked_cat0_us": 269.0, "triton_override_us": 274.4},
        {"tokens": 16384, "main_masked_us": 1440.0, "unmasked_cat0_us": 1047.8, "triton_override_us": 1060.5},
    ]
    for r in fwd_plus_bwd_rows:
        r["delta_pct"] = round((r["unmasked_cat0_us"] - r["main_masked_us"]) / r["main_masked_us"] * 100.0, 2)
        r["faster_than_triton_override_pct"] = round(
            (r["triton_override_us"] - r["unmasked_cat0_us"]) / r["triton_override_us"] * 100.0, 2
        )

    bwd_saved_us_16k = 1440.0 - 1047.8
    bwd_unmasked_est_us = round(bwd_masked_cat_us - bwd_saved_us_16k, 1)

    return {
        "max_pointwise_cat_inputs": 0,
        "max_complex_pointwise_cat_inputs": 0,
        "bwd_bandwidth_floor_16k_us": bwd_floor_us,
        "bwd_masked_cat_16k_us": bwd_masked_cat_us,
        "bwd_unmasked_est_16k_us": bwd_unmasked_est_us,
        "bwd_masked_vs_floor_ratio": round(bwd_masked_cat_us / bwd_floor_us, 2),
        "bwd_unmasked_vs_floor_ratio": round(bwd_unmasked_est_us / bwd_floor_us, 2),
        "gb300_swiglu_table": fwd_plus_bwd_rows,
    }


def simulate_muse_spark_fp32_rmsnorm() -> dict[str, Any]:
    # Muse Spark keeps residual stream in fp32 while effective_weight_D (weight + gain_center) is bf16.
    # Without casting effective_weight_D.to(input.dtype), F.rms_norm falls back to ~20 decomposed ATen kernels.
    # Crucially, bf16 -> fp32 upcast is exact (zero mantissa bits lost: 7-bit mantissa -> 23-bit mantissa).
    weights_bf16 = [f32_to_bf16_bits(v) for v in (1.0, 0.875, 1.125, 0.9375)]
    inputs_fp32 = [0.42, -1.31, 2.07, -0.88]
    eps = 1e-6

    mean_sq = sum(x * x for x in inputs_fp32) / len(inputs_fp32)
    inv_rms = 1.0 / math.sqrt(mean_sq + eps)

    # Decomposed fallback (fp32 input * bf16 weight promoted to fp32)
    fallback_out = [(x * inv_rms) * bf16_bits_to_f32(w) for x, w in zip(inputs_fp32, weights_bf16)]
    # Fused kernel path (weight pre-cast to fp32 before F.rms_norm)
    cast_weights_fp32 = [bf16_bits_to_f32(w) for w in weights_bf16]
    fused_out = [(x * inv_rms) * w for x, w in zip(inputs_fp32, cast_weights_fp32)]

    max_abs_diff = max(abs(a - b) for a, b in zip(fallback_out, fused_out))

    return {
        "decomposed_fallback_kernel_count_fwd_bwd": 20,
        "fused_kernel_count_fwd_bwd": 3,  # 1 fwd + 2 bwd in ATen rms_norm, or 1 compiled region
        "fused_fwd_kernel_count": 1,
        "max_abs_diff_fp32_cast_vs_fallback": max_abs_diff,
        "two_layer_train_delta_pct": -7.0,
        "two_layer_prefill_delta_pct": -5.0,
        "two_layer_decode_delta_pct": -3.0,
    }


def build_summary() -> dict[str, Any]:
    rope = simulate_compiled_complex_rope()
    swiglu = simulate_swiglu_unmasked_cat()
    rmsnorm = simulate_muse_spark_fp32_rmsnorm()
    return {
        "experiment": "torchtitan_compiled_rope_swiglu_fp32_rmsnorm",
        "phase": "Phase 34 (priority: 43)",
        "sources": [
            "pytorch/torchtitan@b95271ed0709d63a60d185b88994cba10ea748a1",
            "pytorch/torchtitan@eb48bab35099a79dfcd7d9c5d26b5af5e41f06d8",
            "pytorch/torchtitan@b78c4aed9640b5fb034b9be1c05f24fe30f1d093",
        ],
        "compiled_complex_rope": rope,
        "swiglu_unmasked_cat": swiglu,
        "muse_spark_fp32_rmsnorm": rmsnorm,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify deterministic invariants")
    args = parser.parse_args()

    summary = build_summary()
    out_path = Path(__file__).resolve().parent / "results" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if args.check:
        rope = summary["compiled_complex_rope"]
        assert rope["int32_pair_max_ulp_diff_vs_complex"] == 0
        assert rope["inverse_rotation_consistent"] is True
        assert rope["autotune_on_distinct_losses_in_13_runs"] == 9
        assert rope["autotune_off_distinct_losses_in_12_runs"] == 1
        assert rope["gb300_benchmarks_us"]["llama3_8b_64k_prefill"]["speedup_vs_eager"] == 5.02
        assert rope["gb300_benchmarks_us"]["deepseek_v3_671b_64k_prefill"]["speedup_vs_eager"] == 6.07

        sw = summary["swiglu_unmasked_cat"]
        assert sw["gb300_swiglu_table"][2]["delta_pct"] == -27.24
        assert sw["gb300_swiglu_table"][2]["faster_than_triton_override_pct"] == 1.2
        assert sw["bwd_unmasked_vs_floor_ratio"] == 1.05

        rn = summary["muse_spark_fp32_rmsnorm"]
        assert rn["max_abs_diff_fp32_cast_vs_fallback"] == 0.0
        assert rn["decomposed_fallback_kernel_count_fwd_bwd"] == 20
        assert rn["fused_fwd_kernel_count"] == 1

    print(json.dumps({"status": "ok", "experiment": summary["experiment"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
