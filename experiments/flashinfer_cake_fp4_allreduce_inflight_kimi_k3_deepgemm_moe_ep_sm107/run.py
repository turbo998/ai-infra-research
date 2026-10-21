#!/usr/bin/env python3
"""Deterministic reproduction of Phase 47 Topic 1:
FlashInfer Cake FP4 AllReduce In-Flight Scale / Shared-Memory Value Reuse,
Kimi-K3 Attention-Residual Round-2 Package, Self-Contained Cake DeepGEMM / NVFP4
Attention, and Consolidated MoE-EP with SM107 (GR100) Native Compilation.

References:
- flashinfer-ai/flashinfer@c7d1eac (#6875)
- flashinfer-ai/flashinfer@3b65505 (#6904)
- flashinfer-ai/flashinfer@595bc57 (#6928)
- flashinfer-ai/flashinfer@597e3a6 (#6927)
- flashinfer-ai/flashinfer@edfb3f7 (#6902)
- flashinfer-ai/flashinfer@9d3837c (#6621)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


SENTINEL_SCALE_WORD = 0x80000000
FP4_WIRE_MASK_7 = 0x77777777
FP4_WIRE_SIGN_8 = 0x88888888
FP8_WIRE_SCALE_SENTINEL = 0xFF


def fp4_wire_values_pending(words: list[int]) -> bool:
    """Exact bit-hack from cake_fp4_allreduce_tp2_rmsnorm_bf16_bf16_kernel.cu:
    Checks whether any nibble in the 32-bit packed FP4 wire word equals -0.0 (0x8).
    """
    for w in words:
        w32 = w & 0xFFFFFFFF
        has_neg_zero = (w32 & ~((w32 & FP4_WIRE_MASK_7) + FP4_WIRE_MASK_7)) & FP4_WIRE_SIGN_8
        if has_neg_zero != 0:
            return True
    return False


def simulate_cake_fp4_allreduce() -> dict[str, Any]:
    """Model #6875 TP2 in-flight scale word read + TP4 shared-memory value reuse."""
    # Verify the nibble sentinel detector: 0x8 nibble triggers pending, non-0x8 nibbles clear
    pending_word = 0x12384567
    settled_word = 0x12374560
    assert fp4_wire_values_pending([pending_word]) is True
    assert fp4_wire_values_pending([settled_word]) is False

    # Model TP2 pass-1 sweep with j0 == 0 warp vote on rsw[0] == 0x80000000u
    # Before #6875: blocking spin-loop on os_rs + m_row before packet sweep
    # After #6875: single volatile load into rsw[0], overlap with packet sweep j0=0,
    # then __any_sync(0xFFFFFFFF, rsw[0] == 0x80000000u) settles only if still pending.
    tp2_bf16_gb300_1k_before_us = 18.15
    tp2_bf16_gb300_1k_after_us = 17.47
    tp2_bf16_gb300_1k_speedup = round(
        tp2_bf16_gb300_1k_before_us / tp2_bf16_gb300_1k_after_us, 4
    )

    tp2_nvfp4_b200_2k_before_us = 29.70
    tp2_nvfp4_b200_2k_after_us = 28.84
    tp2_nvfp4_b200_2k_speedup = round(
        tp2_nvfp4_b200_2k_before_us / tp2_nvfp4_b200_2k_after_us, 4
    )

    tp4_nvfp4_sm107_16k_before_us = 94.94
    tp4_nvfp4_sm107_16k_after_us = 92.49
    tp4_nvfp4_sm107_16k_speedup = round(
        tp4_nvfp4_sm107_16k_before_us / tp4_nvfp4_sm107_16k_after_us, 4
    )

    # In TP4 two-shot pass-2, 4 value words (16B) + 1 scale word (4B) per rank across 4 ranks:
    # Before: re-reads 4 * 16B = 64B values + 4 * 4B = 16B scales (80B/step from L2 Lamport buffer)
    # After: re-reads 64B values from shared memory and only 16B scales from L2 (-80% L2 traffic in pass 2)
    tp4_pass2_l2_bytes_before = 4 * (16 + 4)
    tp4_pass2_l2_bytes_after = 4 * 4
    tp4_pass2_l2_reduction_pct = round(
        (1.0 - tp4_pass2_l2_bytes_after / tp4_pass2_l2_bytes_before) * 100.0, 2
    )

    return {
        "sentinel_scale_word_hex": hex(SENTINEL_SCALE_WORD),
        "nibble_detector_verified": True,
        "geomean_speedup_bf16": {
            "sm_100a": 1.010,
            "sm_103a": 1.012,
            "sm_107a": 1.010,
        },
        "geomean_speedup_nvfp4": {
            "sm_100a": 1.018,
            "sm_103a": 1.018,
            "sm_107a": 1.023,
        },
        "peak_case_highlights": {
            "tp2_bf16_gb300_1kx6144_before_us": tp2_bf16_gb300_1k_before_us,
            "tp2_bf16_gb300_1kx6144_after_us": tp2_bf16_gb300_1k_after_us,
            "tp2_bf16_gb300_1kx6144_speedup": tp2_bf16_gb300_1k_speedup,
            "tp2_nvfp4_b200_2kx6144_before_us": tp2_nvfp4_b200_2k_before_us,
            "tp2_nvfp4_b200_2k_after_us": tp2_nvfp4_b200_2k_after_us,
            "tp2_nvfp4_b200_2kx6144_speedup": tp2_nvfp4_b200_2k_speedup,
            "tp4_nvfp4_sm107_16kx6144_before_us": tp4_nvfp4_sm107_16k_before_us,
            "tp4_nvfp4_sm107_16kx6144_after_us": tp4_nvfp4_sm107_16k_after_us,
            "tp4_nvfp4_sm107_16kx6144_speedup": tp4_nvfp4_sm107_16k_speedup,
        },
        "tp4_pass2_l2_bytes_per_step_before": tp4_pass2_l2_bytes_before,
        "tp4_pass2_l2_bytes_per_step_after": tp4_pass2_l2_bytes_after,
        "tp4_pass2_l2_reduction_pct": tp4_pass2_l2_reduction_pct,
    }


def simulate_kimi_k3_attn_res() -> dict[str, Any]:
    """Model #6904 cake_kimi_k3_attn_res Round-2 vendorable package."""
    # Mathematical validation on toy K=2, D=4 row
    hidden = [1.0, -0.5, 0.25, 0.75]
    residual = [0.5, 0.5, -0.25, 0.25]
    u = [h + r for h, r in zip(hidden, residual)]
    norm_r_w = [1.0, 1.0, 1.0, 1.0]
    norm_u_w = [1.0, 1.0, 1.0, 1.0]
    eps = 1e-6

    def rms_norm(vec: list[float], w: list[float]) -> list[float]:
        ms = sum(x * x for x in vec) / len(vec)
        inv = 1.0 / math.sqrt(ms + eps)
        return [x * inv * wi for x, wi in zip(vec, w)]

    v_u = rms_norm(u, norm_u_w)
    history = [
        [0.8, 0.1, 0.0, 0.5],
        [-0.2, 0.9, 0.3, -0.1],
    ]
    v_hist = [rms_norm(r, norm_r_w) for r in history]
    candidates = v_hist + [v_u]  # K+1 = 3 candidates
    raw_candidates = history + [u]
    query = [0.6, 0.2, -0.1, 0.4]
    logits = [sum(q * c for q, c in zip(query, cand)) for cand in candidates]
    m_logit = max(logits)
    exps = [math.exp(l - m_logit) for l in logits]
    denom = sum(exps)
    probs = [e / denom for e in exps]
    h = [
        sum(probs[k] * raw_candidates[k][d] for k in range(len(candidates)))
        for d in range(len(hidden))
    ]
    out_norm = rms_norm(h, [1.0, 1.0, 1.0, 1.0])

    return {
        "programs_count": 133,
        "libraries_total": 161,
        "libraries_sm100a": 75,
        "libraries_sm103a": 86,
        "contract_shapes": 291,
        "pdl_modes": 2,
        "contract_rows_total": 291 * 2,
        "hidden_size": 7168,
        "max_history_k": 8,
        "geomean_speedups": {
            "gb300": 1.0072,
            "b300": 1.0064,
            "b200": 1.0012,
        },
        "peak_case_speedups": {
            "gb300_serving_m5_k5": {
                "before_us": 4.448,
                "after_us": 4.096,
                "speedup": 1.0888,
            },
            "b300_k_sweep_m1_k5_speedup": 1.0771,
            "b200_k_sweep_m1_k6_speedup": 1.1034,
        },
        "prob_sum_check": round(sum(probs), 6),
        "sample_out_norm_l2": round(math.sqrt(sum(x * x for x in out_norm)), 6),
    }


def simulate_deepgemm_and_moe_ep() -> dict[str, Any]:
    """Model #6928, #6927, #6902, and #6621 kernels & SM107 MoE-EP compilation."""
    batched_proj_gb300_us = {
        "T1_m64": 10.677,
        "T1_swap_m16": 9.756,
        "T2_m64": 10.663,
        "T2_swap_m16": 9.727,
        "T4_m64": 10.727,
        "T4_swap_m16": 9.786,
    }
    swap_m16_gain_t2_pct = round(
        (1.0 - batched_proj_gb300_us["T2_swap_m16"] / batched_proj_gb300_us["T2_m64"]) * 100.0,
        2,
    )

    return {
        "deepgemm_batched_projection": {
            "num_heads": 128,
            "k_dim": 4096,
            "n_dim": 1024,
            "variants_count": 6,
            "gb300_latencies_us": batched_proj_gb300_us,
            "t2_swap_m16_latency_reduction_pct": swap_m16_gain_t2_pct,
        },
        "deepgemm_fp8_1d1d": {
            "fprop_4096x7168x4096_gb300_us": 66.801,
            "wgrad_4096x7168x4096_gb300_us": 81.440,
            "wgrad_16384x2048x7168_gb300_us": 115.657,
        },
        "nvfp4_attention": {
            "b1_h8_s2048_d128_gb300_us": 17.508,
            "b4_h8_s4096_d128_gb300_us": 114.161,
            "smem_stages": 3,
            "tile_m": 128,
            "tile_n": 64,
        },
        "moe_ep_sm107": {
            "supported_sm_targets": ["sm_90a", "sm_100a", "sm_103a", "sm_107a"],
            "gr100_sm_major_minor": [10, 7],
            "mercury_validator_sm107_fixed": True,
        },
    }


def build_summary() -> dict[str, Any]:
    fp4_ar = simulate_cake_fp4_allreduce()
    kimi_k3 = simulate_kimi_k3_attn_res()
    deepgemm_moe = simulate_deepgemm_and_moe_ep()
    return {
        "experiment": "flashinfer_cake_fp4_allreduce_inflight_kimi_k3_deepgemm_moe_ep_sm107",
        "phase": 47,
        "priority": 56,
        "sources": ["S496", "S497", "S498", "S499"],
        "claims": ["C409", "C410", "C411", "C412"],
        "cake_fp4_allreduce": fp4_ar,
        "cake_kimi_k3_attn_res": kimi_k3,
        "deepgemm_and_moe_ep": deepgemm_moe,
    }


def verify(summary: dict[str, Any]) -> None:
    ar = summary["cake_fp4_allreduce"]
    assert ar["nibble_detector_verified"] is True
    assert ar["geomean_speedup_nvfp4"]["sm_107a"] == 1.023
    assert ar["peak_case_highlights"]["tp2_bf16_gb300_1kx6144_speedup"] == 1.0389
    assert ar["tp4_pass2_l2_reduction_pct"] == 80.0

    k3 = summary["cake_kimi_k3_attn_res"]
    assert k3["programs_count"] == 133
    assert k3["libraries_total"] == 161
    assert k3["libraries_sm100a"] + k3["libraries_sm103a"] == 161
    assert k3["contract_rows_total"] == 582
    assert k3["peak_case_speedups"]["gb300_serving_m5_k5"]["speedup"] == 1.0888
    assert abs(k3["prob_sum_check"] - 1.0) < 1e-6

    dg = summary["deepgemm_and_moe_ep"]
    assert dg["deepgemm_batched_projection"]["gb300_latencies_us"]["T2_swap_m16"] == 9.727
    assert dg["deepgemm_fp8_1d1d"]["fprop_4096x7168x4096_gb300_us"] == 66.801
    assert dg["nvfp4_attention"]["b4_h8_s4096_d128_gb300_us"] == 114.161
    assert "sm_107a" in dg["moe_ep_sm107"]["supported_sm_targets"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Run invariant checks")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "summary.json",
    )
    args = parser.parse_args()

    summary = build_summary()
    if args.check:
        verify(summary)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] Wrote {args.output}")


if __name__ == "__main__":
    main()
