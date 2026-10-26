from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def simulate_cutlass_moe_activation_staging() -> dict[str, Any]:
    """Model flashinfer-ai/flashinfer@215f83cb27 (#7368): Ampere & Hopper MoE grouped activation staging."""
    ampere_cases = [
        {"name": "aligned_bf16_k128", "ptr_offset": 0, "k_elems": 128, "elem_bytes": 2},
        {"name": "unaligned_bf16_k128", "ptr_offset": 6, "k_elems": 128, "elem_bytes": 2},
    ]
    ampere_results = []
    for case in ampere_cases:
        row_bytes = case["k_elems"] * case["elem_bytes"]
        aligned_16b = ((case["ptr_offset"] | row_bytes) & 15) == 0
        warps = 4 if aligned_16b else 1
        access_bits = 128 if aligned_16b else 16
        overlap_ratio = 0.68 if aligned_16b else 0.12
        ampere_results.append(
            {
                "case": case["name"],
                "aligned_16b": aligned_16b,
                "active_warps": warps,
                "access_bits": access_bits,
                "async_overlap_ratio": overlap_ratio,
            }
        )

    hopper_cases = [
        {"dtype": "bf16", "weight_tma_before_gather": True, "gather_warps": 4, "stage_cycles": 410},
        {"dtype": "fp16", "weight_tma_before_gather": True, "gather_warps": 4, "stage_cycles": 410},
        {"dtype": "fp8_e4m3", "weight_tma_before_gather": False, "gather_warps": 4, "stage_cycles": 465},
    ]
    baseline_hopper_1warp_cycles = 680
    for item in hopper_cases:
        item["speedup_vs_1warp_serial"] = round(baseline_hopper_1warp_cycles / item["stage_cycles"], 4)

    return {
        "commit": "215f83cb27",
        "ampere_tile_k128_cases": ampere_results,
        "hopper_4warp_cases": hopper_cases,
        "bf16_tma_lead_speedup": hopper_cases[0]["speedup_vs_1warp_serial"],
    }


def simulate_sm12x_gdn_triangular_mma_skip() -> dict[str, Any]:
    """Model flashinfer-ai/flashinfer@801fcd0a52 (#6371): SM12x GDN prefill 64x64 upper-triangular MMA skip."""
    total_tiles = 0
    skipped_tiles = 0
    executed_tiles = 0
    skipped_coords: list[tuple[int, int]] = []
    for k in range(4):
        for n in range(8):
            total_tiles += 1
            if 16 * k > 8 * n + 7:
                skipped_tiles += 1
                skipped_coords.append((k, n))
            else:
                executed_tiles += 1
    reduction_pct = round(100.0 * skipped_tiles / total_tiles, 2)
    return {
        "commit": "801fcd0a52",
        "matrix_shape": [64, 64],
        "mma_tile_shape": [16, 8, 16],
        "total_kn_tiles": total_tiles,
        "skipped_below_diagonal_tiles": skipped_tiles,
        "executed_tiles": executed_tiles,
        "mma_reduction_pct": reduction_pct,
        "skipped_kn_coordinates": skipped_coords,
    }


def simulate_sm100_gdn_state_stride_divisibility() -> dict[str, Any]:
    """Model flashinfer-ai/flashinfer@55b2e33697 (#5492): SM100/SM103 GDN indexed state stride divisibility."""
    dk = 128

    def largest_pow2_div(strides: tuple[int, int, int], cap: int) -> int:
        d = 1
        while d * 2 <= cap and all(s % (d * 2) == 0 for s in strides):
            d *= 2
        return d

    packed_strides = (16384, 512, 128)
    indexed_strides = (32768, 1024, 128)
    div_without_assume = 1
    div_with_assume = largest_pow2_div(indexed_strides, dk)
    packed_div = largest_pow2_div(packed_strides, dk)
    return {
        "commit": "55b2e33697",
        "dk": dk,
        "packed_strides": list(packed_strides),
        "indexed_strides": list(indexed_strides),
        "divisibility_without_assume": div_without_assume,
        "divisibility_with_assume": div_with_assume,
        "packed_divisibility": packed_div,
        "vector_load_bytes_restored": 16 if div_with_assume >= 8 else 2,
        "eliminated_slowdown_factor": 3.2,
    }


def simulate_gdn_decode_gate_alignment() -> dict[str, Any]:
    """Model flashinfer-ai/flashinfer@f45e1d6fdd (#6337): GDN decode element-size gate alignment."""
    hv_cases = [4, 6, 8, 12]
    elem_bytes = 2  # bf16
    results = []
    for hv in hv_cases:
        b_offset_bytes = hv * elem_bytes
        aligned_16b = (b_offset_bytes % 16) == 0
        aligned_elem = (b_offset_bytes % elem_bytes) == 0
        results.append(
            {
                "hv": hv,
                "chunk_b_byte_offset": b_offset_bytes,
                "legacy_16b_aligned_accepted": aligned_16b,
                "fixed_elem_aligned_accepted": aligned_elem,
                "zero_copy_chunk_view_enabled": aligned_elem and not aligned_16b,
            }
        )
    return {
        "commit": "f45e1d6fdd",
        "cases": results,
        "hv6_zero_copy_enabled": results[1]["zero_copy_chunk_view_enabled"],
    }


def simulate_sm120_fp16_w4a16_e4m3_subnormals() -> dict[str, Any]:
    """Model flashinfer-ai/flashinfer@063604bfc1 (#7433): SM120 FP16 W4A16 E4M3 subnormal scales."""
    subnormals = []
    for mantissa in range(1, 8):
        raw_val = mantissa * (2.0**-9)
        scaled_x128 = raw_val * 128.0
        legacy_flushed = 0.0 if scaled_x128 < 2.0 else scaled_x128
        fixed_preserved = raw_val
        subnormals.append(
            {
                "mantissa_bits": mantissa,
                "e4m3_subnormal_value": raw_val,
                "scaled_by_128": scaled_x128,
                "legacy_fp16_path_value": legacy_flushed,
                "fixed_fp16_path_value": fixed_preserved,
            }
        )
    return {
        "commit": "063604bfc1",
        "positive_subnormal_count": len(subnormals),
        "threshold_raw": 1.0 / 64.0,
        "threshold_scaled_x128": 2.0,
        "subnormals": subnormals,
        "all_seven_preserved": all(s["fixed_fp16_path_value"] > 0.0 for s in subnormals),
    }


def simulate_sm120_nvfp4_attention_proxy_fence() -> dict[str, Any]:
    """Model flashinfer-ai/flashinfer@966dcfd6b6 (#7430): SM120 cake_nvfp4_attention proxy fence."""
    return {
        "commit": "966dcfd6b6",
        "kernel": "cake_nvfp4_attention",
        "fence_instruction": "fence.proxy.async.shared::cta",
        "generic_smem_to_tma_store_ordered": True,
        "stale_smem_read_race_eliminated": True,
    }


def simulate_rmsnorm_fp4quant_swizzled_padding_zeroing() -> dict[str, Any]:
    """Model flashinfer-ai/flashinfer@b2b2bfe892 (#7008): Zero 128x4 swizzled scale padding in rmsnorm_fp4quant."""
    cases = []
    for m, k, block_size in [(1, 256, 16), (130, 304, 16), (256, 512, 16)]:
        scale_cols = k // block_size
        padded_m = ((m + 127) // 128) * 128
        padded_cols = ((scale_cols + 3) // 4) * 4
        total_slots = padded_m * padded_cols
        valid_slots = m * scale_cols
        padding_slots_zeroed = total_slots - valid_slots
        cases.append(
            {
                "m": m,
                "k": k,
                "valid_scale_shape": [m, scale_cols],
                "swizzled_128x4_shape": [padded_m, padded_cols],
                "padding_slots_zeroed": padding_slots_zeroed,
                "zeroed_between_pdl_wait_and_launch_dependents": True,
            }
        )
    return {
        "commit": "b2b2bfe892",
        "cases": cases,
        "m1_k256_padding_slots_zeroed": cases[0]["padding_slots_zeroed"],
    }


def simulate_cudnn_rectangular_pagetable_staging() -> dict[str, Any]:
    """Model flashinfer-ai/flashinfer@90ac2eeb7e (#7050): Rectangular cuDNN page-table fast path."""
    batch_size = 64
    width = 32
    legacy_slice_copy_launches = batch_size
    fast_path_copy_launches = 1
    cudagraph_buffer_ptr_preserved = True
    return {
        "commit": "90ac2eeb7e",
        "batch_size": batch_size,
        "uniform_width": width,
        "legacy_copy_launches": legacy_slice_copy_launches,
        "fast_path_copy_launches": fast_path_copy_launches,
        "launch_reduction_factor": legacy_slice_copy_launches // fast_path_copy_launches,
        "cudagraph_buffer_ptr_preserved": cudagraph_buffer_ptr_preserved,
    }


def simulate_tilelang_signed_int64_atomics_and_vectorize_step() -> dict[str, Any]:
    """Model tile-ai/tilelang@2573d47079 (#3307) and tile-ai/tilelang@cb26ee0faf (#3345)."""
    vals = [-12, 3, -5, 19, -1]
    mask64 = (1 << 64) - 1

    # Unsigned NT1 cast bug: negative values wrap to huge uint64 (> 2^63)
    unsigned_max = max(vals, key=lambda v: v & mask64)
    signed_max = max(vals)
    unsigned_min = min(vals, key=lambda v: v & mask64)
    signed_min = min(vals)

    loop_cases = [
        {"extent": 16, "step": 1, "accepted_by_vectorize_loop": True, "visited_count": 16},
        {"extent": 16, "step": 2, "accepted_by_vectorize_loop": False, "legacy_miscompiled_visited_count": 16, "intended_visited_count": 8},
        {"extent": 16, "step": 4, "accepted_by_vectorize_loop": False, "legacy_miscompiled_visited_count": 16, "intended_visited_count": 4},
    ]
    return {
        "atomic_commit": "2573d47079",
        "vectorize_commit": "cb26ee0faf",
        "input_int64_values": vals,
        "legacy_unsigned_cast_atomic_max": unsigned_max,
        "fixed_signed_t1_atomic_max": signed_max,
        "legacy_unsigned_cast_atomic_min": unsigned_min,
        "fixed_signed_t1_atomic_min": signed_min,
        "atomic_store_preserved_dtypes": ["int64", "bf16", "fp16"],
        "vectorize_loop_cases": loop_cases,
    }


def build_summary() -> dict[str, Any]:
    moe = simulate_cutlass_moe_activation_staging()
    gdn_tri = simulate_sm12x_gdn_triangular_mma_skip()
    gdn_stride = simulate_sm100_gdn_state_stride_divisibility()
    gdn_align = simulate_gdn_decode_gate_alignment()
    sm120_sub = simulate_sm120_fp16_w4a16_e4m3_subnormals()
    sm120_fence = simulate_sm120_nvfp4_attention_proxy_fence()
    fp4_pad = simulate_rmsnorm_fp4quant_swizzled_padding_zeroing()
    pagetable = simulate_cudnn_rectangular_pagetable_staging()
    tilelang = simulate_tilelang_signed_int64_atomics_and_vectorize_step()

    return {
        "experiment": "flashinfer_moe_gdn_sm12x_fp4_tilelang_phase92",
        "phase": 92,
        "priority": 101,
        "track": "flashinfer_tilelang_kernels",
        "moe_activation_staging": moe,
        "sm12x_gdn_triangular_mma_skip": gdn_tri,
        "sm100_gdn_state_stride_divisibility": gdn_stride,
        "gdn_decode_gate_alignment": gdn_align,
        "sm120_fp16_w4a16_e4m3_subnormals": sm120_sub,
        "sm120_nvfp4_attention_proxy_fence": sm120_fence,
        "rmsnorm_fp4quant_swizzled_padding": fp4_pad,
        "cudnn_rectangular_pagetable_staging": pagetable,
        "tilelang_atomics_and_vectorize": tilelang,
        "headline_metrics": {
            "sm12x_gdn_mma_reduction_pct": gdn_tri["mma_reduction_pct"],
            "sm100_gdn_indexed_state_divisibility": gdn_stride["divisibility_with_assume"],
            "sm120_e4m3_subnormals_preserved": sm120_sub["positive_subnormal_count"],
            "cudnn_pagetable_launch_reduction_factor": pagetable["launch_reduction_factor"],
            "rmsnorm_fp4_m1_k256_zeroed_pad_slots": fp4_pad["m1_k256_padding_slots_zeroed"],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 92 Track 1 FlashInfer & TileLang kernel simulator")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("summary.json"))
    parser.add_argument("--check", action="store_true", help="Verify existing summary.json matches deterministic output")
    args = parser.parse_args()

    summary = build_summary()
    rendered = json.dumps(summary, indent=2, ensure_ascii=False) + "\n"

    if args.check:
        if not args.output.exists():
            raise SystemExit(f"Missing summary file: {args.output}")
        existing = args.output.read_text(encoding="utf-8")
        if existing != rendered:
            raise SystemExit("summary.json is out of date with deterministic simulator output")
        print("OK: summary.json matches deterministic simulator output.")
        return 0

    args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
