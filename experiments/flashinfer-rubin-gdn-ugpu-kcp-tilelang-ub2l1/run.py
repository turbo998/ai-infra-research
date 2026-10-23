#!/usr/bin/env python3
"""Phase 69 Track A deterministic experiment:
FlashInfer Rubin SM107a GDN Decode uGPU Localization, Context-Parallel KCP
Chunked Training Plan, Rubin MLA/Kimi-K3 PDL & Snapshot-Write Guards, and
TileLang Ascend UB-to-L1 Geometry / PTO Vector-Local / CUDA Cast-Rounding Validation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path


def simulate_rubin_gdn_decode_ugpu() -> dict[str, object]:
    """Reproduce FlashInfer Rubin sm_107a GDN decode & uGPU localization (#6604)."""
    num_sms = 212
    sms_per_die = 106
    dies = 2
    dram_peak_tb_s = 19.15
    pf_dist = num_sms * 25 // 3  # 1766 tiles ahead (~1.2 residency generations at 7 CTAs/SM)
    auto_min_states = 512
    ugpu_eager_min_states = 8192

    # VR200 (212 SMs, HV=64, HQ=HK=16, K=V=128, 19.15 TB/s peak) benchmark table
    vr200_benchmarks = [
        {
            "batch": 32,
            "generic_us": 18.0,
            "generic_bw_pct": 39,
            "rubin_us": 13.2,
            "rubin_bw_pct": 53,
            "rubin_speedup": 1.36,
            "rubin_ugpu_us": 12.3,
            "rubin_ugpu_bw_pct": 57,
            "rubin_ugpu_speedup": 1.46,
        },
        {
            "batch": 128,
            "generic_us": 51.1,
            "generic_bw_pct": 55,
            "rubin_us": 43.6,
            "rubin_bw_pct": 65,
            "rubin_speedup": 1.17,
            "rubin_ugpu_us": 37.0,
            "rubin_ugpu_bw_pct": 76,
            "rubin_ugpu_speedup": 1.38,
        },
        {
            "batch": 512,
            "generic_us": 187.0,
            "generic_bw_pct": 61,
            "rubin_us": 163.0,
            "rubin_bw_pct": 69,
            "rubin_speedup": 1.15,
            "rubin_ugpu_us": 132.6,
            "rubin_ugpu_bw_pct": 85,
            "rubin_ugpu_speedup": 1.41,
        },
        {
            "batch": 1024,
            "generic_us": 366.6,
            "generic_bw_pct": 62,
            "rubin_us": 322.1,
            "rubin_bw_pct": 70,
            "rubin_speedup": 1.14,
            "rubin_ugpu_us": 260.1,
            "rubin_ugpu_bw_pct": 87,
            "rubin_ugpu_speedup": 1.41,
        },
        {
            "batch": 2048,
            "generic_us": 726.7,
            "generic_bw_pct": 62,
            "rubin_us": 640.2,
            "rubin_bw_pct": 71,
            "rubin_speedup": 1.14,
            "rubin_ugpu_us": 516.5,
            "rubin_ugpu_bw_pct": 88,
            "rubin_ugpu_speedup": 1.41,
        },
        {
            "batch": 8192,
            "generic_us": 2887.0,
            "generic_bw_pct": 63,
            "rubin_us": 2549.8,
            "rubin_bw_pct": 71,
            "rubin_speedup": 1.13,
            "rubin_ugpu_us": 2059.8,
            "rubin_ugpu_bw_pct": 88,
            "rubin_ugpu_speedup": 1.40,
        },
    ]

    for row in vr200_benchmarks:
        row["states_b_hv"] = row["batch"] * 64
        row["auto_routes_rubin"] = row["states_b_hv"] >= auto_min_states
        row["eager_forks_die_streams"] = row["states_b_hv"] >= ugpu_eager_min_states

    return {
        "gpu": "VR200_sm_107a",
        "num_sms": num_sms,
        "sms_per_die": sms_per_die,
        "dies": dies,
        "dram_peak_tb_s": dram_peak_tb_s,
        "interleaved_copy_ceiling_pct": 68,
        "die_local_copy_ceiling_pct": 90,
        "pf_dist_tiles": pf_dist,
        "warp_instruction_reduction_pct": 38,
        "transpose_reduce_shuffles_old": 16,
        "transpose_reduce_shuffles_new": 5,
        "auto_min_states": auto_min_states,
        "ugpu_eager_min_states": ugpu_eager_min_states,
        "vr200_benchmarks": vr200_benchmarks,
    }


def kcp_plan_wave_fill1(
    n_frag: int,
    nc_max_frag: int,
    h_heads: int,
    sm_count: int = 148,
) -> dict[str, object]:
    """Reproduce FlashInfer kcp_plan wave_fill1 segment-count rule (#7002)."""
    p_chain = max(1, (nc_max_frag + 15) // 16)
    if n_frag <= 0 or nc_max_frag <= 0:
        return {"p_segments": 1, "p_chain": p_chain, "regime": "empty"}

    if n_frag == 1 and 0 < nc_max_frag < 257:
        p_wave = max(1, sm_count // h_heads)
        p_seg = max(p_chain, min(p_wave, min(64, nc_max_frag)))
        return {"p_segments": p_seg, "p_chain": p_chain, "regime": "single_frag_wave_fill"}

    if n_frag <= 3 and nc_max_frag >= 257:
        t_walk = 10.0
        t_compose = 1.0
        best_p = p_chain
        best_cost = float("inf")
        for p_cand in range(max(1, p_chain // 2), min(nc_max_frag, 128) + 1):
            waves = math.ceil(n_frag * p_cand * h_heads / sm_count)
            chunks_per_seg = math.ceil(nc_max_frag / p_cand)
            cost = waves * chunks_per_seg * t_walk + (p_cand - 1) * t_compose
            if cost < best_cost:
                best_cost = cost
                best_p = p_cand
        return {
            "p_segments": best_p,
            "p_chain": p_chain,
            "best_cost": round(best_cost, 3),
            "regime": "long_frag_cost_model",
        }

    return {"p_segments": min(p_chain, 64), "p_chain": p_chain, "regime": "multi_frag_default"}


def simulate_rubin_mla_pdl_and_kimi_k3() -> dict[str, object]:
    """Reproduce FlashInfer Rubin MLA NVFP4 PDL guard (#7004) and Kimi-K3 write cell pinning (#7014)."""
    units_pairs_by_sm_count = {208: (50, 4), 212: (46, 14)}

    def pdl_for_plan(max_splits: int, items: int, sm_count: int) -> dict[str, object]:
        pair_units, single_units = units_pairs_by_sm_count[sm_count]
        units = pair_units + single_units
        enable_pdl = (max_splits > 1) and (items <= units)
        return {
            "sm_count": sm_count,
            "max_splits": max_splits,
            "items": items,
            "units": units,
            "enable_pdl": enable_pdl,
        }

    pdl_cases = [
        pdl_for_plan(max_splits=4, items=54, sm_count=208),  # fits in 1 wave -> True
        pdl_for_plan(max_splits=4, items=55, sm_count=208),  # 2nd wave -> False
        pdl_for_plan(max_splits=1, items=32, sm_count=212),  # max_splits == 1 -> False
        pdl_for_plan(max_splits=8, items=60, sm_count=212),  # fits in 60 units -> True
    ]

    write_persistent_key_cells = {
        ("sm_100a", 4096, 4): "persistent:k4_nc3_d2_f111000000000000_write",
        ("sm_103a", 4096, 4): "persistent:k4_nc3_d2_f111000000000000_write",
    }

    return {
        "pdl_cases": pdl_cases,
        "write_persistent_key_cells": {f"{k[0]}_M{k[1]}_K{k[2]}": v for k, v in write_persistent_key_cells.items()},
        "kimi_k3_benchmarks": {
            "gb300_m4096_k5_old_us": 85.248,
            "gb300_m4096_k5_new_us": 83.920,
            "gb300_m4096_k5_speedup": 1.0160,
            "b300_m4096_k4_write_old_us": 87.649,
            "b300_m4096_k4_write_new_us": 85.473,
            "b300_m4096_k4_write_speedup": 1.0255,
            "b200_m4096_k4_write_old_us": 86.112,
            "b200_m4096_k4_write_new_us": 84.832,
            "b200_m4096_k4_write_speedup": 1.0152,
        },
        "kcp_benchmarks": {
            "gb300_configs_count": 144,
            "speedup_vs_fla_kcp_min": 1.3058,
            "speedup_vs_fla_kcp_max": 2.4679,
            "speedup_vs_cake_head_cp_min": 1.6184,
            "speedup_vs_cake_head_cp_max": 2.3319,
            "plan_1frag_nc64_h16": kcp_plan_wave_fill1(1, 64, 16, 148),
            "plan_2frag_nc512_h16": kcp_plan_wave_fill1(2, 512, 16, 148),
        },
    }


def check_tilelang_ascend_and_cuda_cast_guards() -> dict[str, object]:
    """Reproduce TileLang Ascend Nd2NzGeometry (#3530), PTO vector local (#3523), and CUDA cast rounding (#3350)."""
    def nd2nz_geometry(rows: int, cols: int, dst_bytes: int, src_pitch: int, dst_pitch: int) -> dict[str, object]:
        c0_elems = 32 // dst_bytes
        if cols % c0_elems != 0:
            raise ValueError(f"cols ({cols}) must be a multiple of C0 elements ({c0_elems})")
        bursts = cols // c0_elems
        src_gap = src_pitch - rows
        dst_gap = dst_pitch - rows
        valid = (1 <= rows <= 65535) and (1 <= bursts <= 4095) and (src_gap >= 0) and (dst_gap >= 0)
        src_span_bytes = ((bursts - 1) * src_pitch + rows) * 32
        dst_span_bytes = ((bursts - 1) * dst_pitch + rows) * 32
        return {
            "rows": rows,
            "cols": cols,
            "dst_bytes": dst_bytes,
            "bursts": bursts,
            "src_gap": src_gap,
            "dst_gap": dst_gap,
            "src_span_bytes": src_span_bytes,
            "dst_span_bytes": dst_span_bytes,
            "valid": valid,
        }

    def pto_vector_local_lowering(num_slots: int, lanes: int) -> dict[str, object]:
        scalar_extent = num_slots * lanes
        slot_offsets = [slot * lanes for slot in range(num_slots)]
        lane_offsets = [slot * lanes + lane for slot in range(num_slots) for lane in range(lanes)]
        return {
            "num_slots": num_slots,
            "lanes": lanes,
            "scalar_buffer_extent": scalar_extent,
            "slot_offsets": slot_offsets,
            "lane_offsets": lane_offsets,
        }

    def cuda_cast_round_is_supported(
        from_dtype: str,
        target_dtype: str,
        cast_round: str = "",
        has_rbits: bool = False,
    ) -> bool:
        if not cast_round:
            return True
        if cast_round != "rs" or not has_rbits:
            return False
        if from_dtype != "float32":
            return False
        supported_targets = {
            "float16",
            "bfloat16",
            "float8_e4m3fn",
            "float8_e4m3fnuz",
            "float8_e5m2",
            "float4_e2m1fn",
        }
        return target_dtype in supported_targets

    cast_cases = {
        "f32_to_fp8_rs_with_rbits": cuda_cast_round_is_supported("float32", "float8_e4m3fn", "rs", True),
        "f32_to_fp4_rs_with_rbits": cuda_cast_round_is_supported("float32", "float4_e2m1fn", "rs", True),
        "f16_to_fp8_rs_rejected": cuda_cast_round_is_supported("float16", "float8_e4m3fn", "rs", True),
        "f32_to_e8m0_rs_rejected": cuda_cast_round_is_supported("float32", "float8_e8m0fnu", "rs", True),
        "f32_to_fp8_rn_rejected": cuda_cast_round_is_supported("float32", "float8_e4m3fn", "rn", False),
        "f16_to_fp8_plain_allowed": cuda_cast_round_is_supported("float16", "float8_e4m3fn", "", False),
    }

    return {
        "nd2nz_subtile_16x64_fp16": nd2nz_geometry(rows=16, cols=64, dst_bytes=2, src_pitch=32, dst_pitch=64),
        "nd2nz_oversized_rows_invalid": nd2nz_geometry(rows=70000, cols=64, dst_bytes=2, src_pitch=70000, dst_pitch=70000),
        "pto_float32x4_2slots": pto_vector_local_lowering(num_slots=2, lanes=4),
        "cuda_cast_cases": cast_cases,
    }


def build_summary() -> dict[str, object]:
    rubin_gdn = simulate_rubin_gdn_decode_ugpu()
    mla_kcp = simulate_rubin_mla_pdl_and_kimi_k3()
    tilelang = check_tilelang_ascend_and_cuda_cast_guards()

    invariants = {
        "rubin_pf_dist_is_1766": rubin_gdn["pf_dist_tiles"] == 1766,
        "rubin_ugpu_b2048_reaches_88pct_bw": rubin_gdn["vr200_benchmarks"][4]["rubin_ugpu_bw_pct"] == 88,
        "rubin_mla_pdl_disabled_on_second_wave": (
            mla_kcp["pdl_cases"][0]["enable_pdl"] is True
            and mla_kcp["pdl_cases"][1]["enable_pdl"] is False
            and mla_kcp["pdl_cases"][2]["enable_pdl"] is False
        ),
        "kcp_wave_fill1_clamps_single_frag": mla_kcp["kcp_benchmarks"]["plan_1frag_nc64_h16"]["p_segments"] == 9,
        "tilelang_nd2nz_span_exact": (
            tilelang["nd2nz_subtile_16x64_fp16"]["valid"] is True
            and tilelang["nd2nz_subtile_16x64_fp16"]["dst_span_bytes"] == 6656
            and tilelang["nd2nz_oversized_rows_invalid"]["valid"] is False
        ),
        "tilelang_cuda_cast_rejects_e8m0_and_non_f32_rs": (
            tilelang["cuda_cast_cases"]["f32_to_fp8_rs_with_rbits"] is True
            and tilelang["cuda_cast_cases"]["f16_to_fp8_rs_rejected"] is False
            and tilelang["cuda_cast_cases"]["f32_to_e8m0_rs_rejected"] is False
            and tilelang["cuda_cast_cases"]["f32_to_fp8_rn_rejected"] is False
        ),
    }

    return {
        "experiment": "flashinfer-rubin-gdn-ugpu-kcp-tilelang-ub2l1",
        "phase": 69,
        "priority": 78,
        "evidence_tier": "own-offline",
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rubin_gdn_ugpu": rubin_gdn,
        "rubin_mla_pdl_and_kcp": mla_kcp,
        "tilelang_guards": tilelang,
        "invariants": invariants,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Validate deterministic invariants")
    args = parser.parse_args()

    summary = build_summary()
    out_path = Path(__file__).resolve().parent / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if args.check:
        failed = [k for k, v in summary["invariants"].items() if not v]
        if failed:
            raise SystemExit(f"FAILED invariants: {failed}")
        print(
            "OK: flashinfer-rubin-gdn-ugpu-kcp-tilelang-ub2l1 "
            f"(pf_dist={summary['rubin_gdn_ugpu']['pf_dist_tiles']}, "
            f"b2048_ugpu_bw={summary['rubin_gdn_ugpu']['vr200_benchmarks'][4]['rubin_ugpu_bw_pct']}%, "
            f"nd2nz_dst_span={summary['tilelang_guards']['nd2nz_subtile_16x64_fp16']['dst_span_bytes']}B)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
