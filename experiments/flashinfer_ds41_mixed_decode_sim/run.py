#!/usr/bin/env python3
"""Deterministic simulator for FlashInfer Frost DeepSeek-V4.1 Mixed MXFP8/FP4 Decode
and CUDA Green Context SM Partitioning.

Models:
1. Non-interleaved 64-token page memory layout (`cache.py`):
   - `swa_mxfp8` (`[pages, 64, 1, 528]`): 512B E4M3 data + 16B E8M0 power-of-2 group-32 scale
     (`_power2(amax)` where `value = amax / 448.0` rounded up to next power of 2).
   - `main_kv_fp4` (`[pages, 64, 1, 288]`): 256B E2M1x2 packed data + 32B E4M3 group-16 scale
     (`scale8 = max(amax, 6 * 2^-9) / 6`), comparing round-to-nearest division (`div_rn`) vs
     reciprocal multiplication (`x * (1.0 / scale)`) on E2M1 halfway boundaries.
2. Adaptive split / Warp-Specialized (WS) / persistent-CTA planner (`decode.py`):
   - `desired = 10 if B<=4 else 8 if B<=16 else 4 if B<=32 else 2 if B<=64 else 1`
   - `split = min(desired, (128 + ck) // 64)`
   - `dequant_warps = 16, phase_chunks = 2` when `ck == 512 and B >= 128`
   - `waves = ceil(B / sm_count)`, `persistent_ctas = ceil(B / waves)` when `B > sm_count`
3. Sink-exclusive natural-log LSE (`lse`) vs final sink normalization (`BlackwellV41MixedCacheDecode`).
4. CUDA Green Context SM resource alignment (`get_sm_count_constraint`, `split_resource_by_sm_count`).
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from pathlib import Path
from typing import Any

RESULTS_PATH = Path(__file__).resolve().parent / "results.json"

E2M1_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def f32(val: float) -> float:
    """Round float to IEEE-754 binary32 precision."""
    return struct.unpack("<f", struct.pack("<f", float(val)))[0]


def power2_e8m0_scale(amax: float) -> tuple[float, int]:
    """Replicate `_power2(amax)` in `flashinfer/experimental/deepseek_v41/cache.py`."""
    effective_amax = max(float(amax), 1e-4)
    val = f32(effective_amax * f32(1.0 / 448.0))
    bits = struct.unpack("<I", struct.pack("<f", val))[0]
    exponent = (bits >> 23) + (1 if (bits & 0x7FFFFF) != 0 else 0)
    scale_bits = (exponent & 0xFF) << 23
    scale_f32 = struct.unpack("<f", struct.pack("<I", scale_bits))[0]
    return scale_f32, exponent & 0xFF


def quantize_e4m3_approx(val: float) -> float:
    """Approximate FP8 E4M3fn rounding (1 sign, 4 exp, 3 mantissa, max 448.0)."""
    if val == 0.0:
        return 0.0
    sign = -1.0 if val < 0 else 1.0
    aval = min(abs(val), 448.0)
    min_subnormal = 2.0**-9
    if aval < min_subnormal * 0.5:
        return 0.0
    exp = math.floor(math.log2(max(aval, 2.0**-6)))
    exp = max(-6, min(8, exp))
    step = (2.0**exp) / 8.0
    quant = round(aval / step) * step
    return sign * min(quant, 448.0)


def cvt_rn_satfinite_e2m1(val: float) -> float:
    """Round-to-nearest-even saturation to FP4 E2M1 levels."""
    sign = -1.0 if val < 0.0 else 1.0
    aval = min(abs(val), 6.0)
    best_level = 0.0
    best_diff = float("inf")
    for idx, lvl in enumerate(E2M1_LEVELS):
        diff = abs(aval - lvl)
        if diff < best_diff - 1e-12:
            best_diff = diff
            best_level = lvl
        elif abs(diff - best_diff) <= 1e-12:
            # Round to even index on exact halfway ties
            if idx % 2 == 0:
                best_level = lvl
    return sign * best_level


def compute_non_interleaved_page_offsets(
    slot: int, fp4: bool, page_size: int = 64
) -> dict[str, int]:
    """Replicate page address arithmetic in `cache.py::_quantize_cache`."""
    width = 256 if fp4 else 512
    group = 16 if fp4 else 32
    sf = 512 // group
    token_stride = width + sf
    page_idx = slot // page_size
    in_page_slot = slot % page_size
    base = page_idx * page_size * token_stride
    data_base = base + in_page_slot * width
    scale_base = base + page_size * width + in_page_slot * sf
    return {
        "slot": slot,
        "page_idx": page_idx,
        "in_page_slot": in_page_slot,
        "width_bytes": width,
        "scale_bytes": sf,
        "token_stride_bytes": token_stride,
        "page_bytes": page_size * token_stride,
        "base_offset": base,
        "data_base_offset": data_base,
        "scale_base_offset": scale_base,
    }


def simulate_cache_quantization_and_layout() -> dict[str, Any]:
    """Compare swa_mxfp8 vs main_kv_fp4 memory footprint and div_rn vs reciprocal accuracy."""
    swa_slot0 = compute_non_interleaved_page_offsets(0, fp4=False)
    swa_slot63 = compute_non_interleaved_page_offsets(63, fp4=False)
    fp4_slot0 = compute_non_interleaved_page_offsets(0, fp4=True)
    fp4_slot63 = compute_non_interleaved_page_offsets(63, fp4=True)

    # Test halfway perturbation between div_rn(x, scale) and x * f32(1.0 / scale)
    # E2M1 halfway boundaries are 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0
    halfway_multipliers = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
    candidate_scales = [
        quantize_e4m3_approx(0.15 + 0.07 * i) for i in range(1, 32)
    ]
    perturbation_cases = 0
    total_cases = 0
    sample_perturbations: list[dict[str, float]] = []

    for scale in candidate_scales:
        if scale <= 0:
            continue
        inv_scale = f32(1.0 / f32(scale))
        for hm in halfway_multipliers:
            # Construct x just above or on the boundary in FP32
            x = f32(hm * scale)
            q_div_rn = f32(x / f32(scale))
            q_recip = f32(x * inv_scale)
            e2m1_div = cvt_rn_satfinite_e2m1(q_div_rn)
            e2m1_rec = cvt_rn_satfinite_e2m1(q_recip)
            total_cases += 1
            if e2m1_div != e2m1_rec:
                perturbation_cases += 1
                if len(sample_perturbations) < 3:
                    sample_perturbations.append(
                        {
                            "scale_e4m3": round(scale, 6),
                            "x_f32": round(x, 6),
                            "div_rn_quotient": round(q_div_rn, 8),
                            "recip_quotient": round(q_recip, 8),
                            "e2m1_div_rn": e2m1_div,
                            "e2m1_recip": e2m1_rec,
                        }
                    )

    # Compare per-token bytes: BF16 (1024B) vs swa_mxfp8 (528B) vs main_kv_fp4 (288B)
    # And hybrid window (128 tokens MXFP8) + global (512 tokens FP4) vs full BF16 (640 tokens)
    bf16_640_bytes = 640 * 512 * 2
    frost_hybrid_640_bytes = 128 * 528 + 512 * 288
    compression_ratio_vs_bf16 = bf16_640_bytes / frost_hybrid_640_bytes

    return {
        "swa_mxfp8_layout": {
            "slot_0": swa_slot0,
            "slot_63": swa_slot63,
            "data_region_end_bytes": 64 * 512,
            "scale_region_end_bytes": 64 * 528,
        },
        "main_kv_fp4_layout": {
            "slot_0": fp4_slot0,
            "slot_63": fp4_slot63,
            "data_region_end_bytes": 64 * 256,
            "scale_region_end_bytes": 64 * 288,
        },
        "halfway_perturbation_analysis": {
            "total_boundary_tests": total_cases,
            "reciprocal_perturbation_count": perturbation_cases,
            "perturbation_rate_pct": round(100.0 * perturbation_cases / max(total_cases, 1), 2),
            "examples": sample_perturbations,
        },
        "hybrid_kv_memory_efficiency": {
            "bf16_640_tokens_bytes": bf16_640_bytes,
            "frost_hybrid_128swa_512fp4_bytes": frost_hybrid_640_bytes,
            "compression_ratio_vs_bf16": round(compression_ratio_vs_bf16, 4),
            "bandwidth_saving_pct": round(
                100.0 * (1.0 - frost_hybrid_640_bytes / bf16_640_bytes), 2
            ),
        },
    }


def plan_ds41_decode_schedule(
    batch: int,
    compressed_k: int,
    sm_count: int = 148,
    max_pool_bytes: int = 1 << 30,
) -> dict[str, Any]:
    """Replicate the exact decode plan heuristic in `flashinfer/experimental/deepseek_v41/decode.py`."""
    desired = (
        10
        if batch <= 4
        else 8
        if batch <= 16
        else 4
        if batch <= 32
        else 2
        if batch <= 64
        else 1
    )
    split = min(desired, (128 + compressed_k) // 64)
    dequant_warps, phase_chunks, narrow_offsets = 8, 8, False
    if compressed_k == 512 and batch >= 128:
        dequant_warps, phase_chunks = 16, 2
    elif compressed_k == 512 and batch >= 64 and max_pool_bytes < (1 << 31):
        phase_chunks, narrow_offsets = 2, True

    persistent_ctas = 0
    waves = 1
    if dequant_warps == 16:
        if batch > sm_count:
            waves = (batch + sm_count - 1) // sm_count
            persistent_ctas = (batch + waves - 1) // waves

    use_ws = dequant_warps == 16 and not (split > 1)
    compact_metadata = bool(persistent_ctas) and max_pool_bytes < (1 << 35)
    workspace_bytes = batch * 64 * split * 513 * 4 if split > 1 else 0

    # Tail wave efficiency: balanced persistent CTAs vs naive full-SM grid
    naive_waves = (batch + sm_count - 1) // sm_count
    naive_last_wave_ctas = batch - (naive_waves - 1) * sm_count
    naive_grid_efficiency = batch / (naive_waves * sm_count)
    effective_ctas = persistent_ctas if persistent_ctas > 0 else min(batch * split, sm_count)
    balanced_efficiency = (
        batch / (waves * persistent_ctas) if persistent_ctas > 0 else naive_grid_efficiency
    )

    return {
        "batch": batch,
        "compressed_k": compressed_k,
        "desired_split": desired,
        "split": split,
        "dequant_warps": dequant_warps,
        "phase_chunks": phase_chunks,
        "narrow_offsets": narrow_offsets,
        "use_ws": use_ws,
        "waves": waves,
        "persistent_ctas": persistent_ctas,
        "compact_metadata": compact_metadata,
        "workspace_bytes": workspace_bytes,
        "naive_last_wave_ctas": naive_last_wave_ctas,
        "naive_grid_utilization_pct": round(100.0 * naive_grid_efficiency, 2),
        "balanced_grid_utilization_pct": round(
            100.0 * min(1.0, balanced_efficiency), 2
        ),
        "effective_ctas": effective_ctas,
    }


def simulate_sink_exclusive_lse() -> dict[str, Any]:
    """Verify sink-exclusive public LSE contract and final sink normalization."""
    swa_logits = [-1.2, 0.5, 1.8, -0.4]
    global_logits = [0.2, 2.1, -0.7, 1.1]
    sink_logit = 3.0  # Strong attention sink

    all_real_logits = swa_logits + global_logits
    max_real = max(all_real_logits)
    sum_exp_real = sum(math.exp(x - max_real) for x in all_real_logits)
    public_lse_natural = max_real + math.log(sum_exp_real)

    # Split-K intermediate log2 LSE for SWA split and Global split
    max_swa = max(swa_logits)
    lse_swa_log2 = (max_swa + math.log(sum(math.exp(x - max_swa) for x in swa_logits))) * math.log2(math.e)
    max_glb = max(global_logits)
    lse_glb_log2 = (max_glb + math.log(sum(math.exp(x - max_glb) for x in global_logits))) * math.log2(math.e)

    # Merge split intermediates in log2 space and convert back to natural log
    max_lse_log2 = max(lse_swa_log2, lse_glb_log2)
    merged_lse_log2 = max_lse_log2 + math.log2(
        2.0 ** (lse_swa_log2 - max_lse_log2) + 2.0 ** (lse_glb_log2 - max_lse_log2)
    )
    merged_lse_natural = merged_lse_log2 / math.log2(math.e)

    # Final output normalization includes sink_logit once at the end
    total_denom_with_sink = math.exp(public_lse_natural) + math.exp(sink_logit)
    real_attention_mass = math.exp(public_lse_natural) / total_denom_with_sink
    sink_attention_mass = math.exp(sink_logit) / total_denom_with_sink

    return {
        "public_lse_natural": round(public_lse_natural, 6),
        "merged_split_lse_natural": round(merged_lse_natural, 6),
        "lse_merge_error": abs(public_lse_natural - merged_lse_natural),
        "sink_logit": sink_logit,
        "real_tokens_attention_mass": round(real_attention_mass, 6),
        "sink_attention_mass": round(sink_attention_mass, 6),
        "mass_conservation": round(real_attention_mass + sink_attention_mass, 6),
    }


def simulate_green_ctx_partitioning(
    total_sms: int = 148, requested_sm_counts: list[int] | None = None
) -> dict[str, Any]:
    """Replicate `flashinfer/green_ctx.py` SM count constraints for SM90/SM100 `(8, 8)`."""
    if requested_sm_counts is None:
        requested_sm_counts = [20, 36, 64]
    min_sm, granularity = 8, 8  # major >= 9 constraint from get_sm_count_constraint
    remaining = total_sms
    allocated_groups: list[dict[str, int]] = []
    for req in requested_sm_counts:
        aligned = max(min_sm, ((req + granularity - 1) // granularity) * granularity)
        if aligned > remaining:
            raise ValueError(f"Insufficient SMs: requested {aligned}, remaining {remaining}")
        remaining -= aligned
        allocated_groups.append(
            {
                "requested_sms": req,
                "aligned_sms": aligned,
                "remaining_sms_after_split": remaining,
            }
        )
    return {
        "total_sms": total_sms,
        "alignment_constraint": {"min_sm": min_sm, "granularity": granularity},
        "partitions": allocated_groups,
        "unallocated_remaining_sms": remaining,
    }


def run_simulation() -> dict[str, Any]:
    cache_sim = simulate_cache_quantization_and_layout()
    schedule_Sweep = [
        plan_ds41_decode_schedule(batch=b, compressed_k=512, sm_count=148)
        for b in (1, 4, 16, 32, 64, 128, 160, 256)
    ]
    lse_sim = simulate_sink_exclusive_lse()
    green_ctx_sim = simulate_green_ctx_partitioning(148, [20, 36, 64])
    return {
        "Metadata": {
            "experiment": "flashinfer_ds41_mixed_decode_sim",
            "upstream_repo": "flashinfer-ai/flashinfer",
            "upstream_commit": "8dc1163fbfbec7098a2418276fd04bc1fa1aeaa5",
            "target_arch": "NVIDIA Blackwell SM100",
        },
        "cache_quantization_and_layout": cache_sim,
        "decode_schedule_sweep": schedule_Sweep,
        "sink_exclusive_lse": lse_sim,
        "green_ctx_partitioning": green_ctx_sim,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate deterministic results against committed results.json",
    )
    args = parser.parse_args()

    results = run_simulation()
    formatted = json.dumps(results, indent=2, sort_keys=True) + "\n"

    if args.check:
        if not RESULTS_PATH.exists():
            sys.stderr.write(f"Missing {RESULTS_PATH}\n")
            return 1
        existing = RESULTS_PATH.read_text(encoding="utf-8")
        if existing != formatted:
            sys.stderr.write("results.json mismatch for flashinfer_ds41_mixed_decode_sim\n")
            return 1
        print("flashinfer_ds41_mixed_decode_sim: --check PASSED")
        return 0

    RESULTS_PATH.write_text(formatted, encoding="utf-8")
    print(f"Wrote {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
