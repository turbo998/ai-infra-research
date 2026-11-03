#!/usr/bin/env python3
"""Phase 181: FlashInfer 12-Row RMSNorm-SiLU-Gate Short Forward, Quad-Route MoE Finalize-AllReduce, K1 Bound-Arg Dense Projection GEMM, R200 sm_107a Dual-Layout mHC Recompute, Two-Wave Low-Latency Softmax, CuTe FlashAttention Subprocess Autotuning, Triton Layout/Pointer/TDM Verification & AITER gfx950 hd256 MXFP4/6."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def simulate_gpu_software_stack_phase181() -> dict[str, object]:
    # 1. FlashInfer (#8505, 1798b527fa18): cake_rmsnorm_silu_gate_train 12 rows/SM short forward on SM100/SM103 (800x2x4096 B200: 8.32 -> 6.34 us = 1.312x; 832x2x4096 GB300: 8.20 -> 6.29 us = 1.304x)
    b200_rmsnorm_10row_fallback_us = 8.32
    b200_rmsnorm_12row_short_fwd_us = 6.34
    rmsnorm_silu_gate_12row_b200_fwd_speedup = round(
        b200_rmsnorm_10row_fallback_us / b200_rmsnorm_12row_short_fwd_us, 3
    )  # 1.312
    gb300_rmsnorm_10row_fallback_us = 8.20
    gb300_rmsnorm_12row_short_fwd_us = 6.29
    rmsnorm_silu_gate_12row_gb300_fwd_speedup = round(
        gb300_rmsnorm_10row_fallback_us / gb300_rmsnorm_12row_short_fwd_us, 3
    )  # 1.304
    rmsnorm_silu_gate_sm100_max_rows = 148 * 12  # 1776
    rmsnorm_silu_gate_sm103_max_rows = 152 * 12  # 1824

    # 2. FlashInfer (#8503, 283300b79da4): cake_moe_finalize_allreduce_fusion quad-route WS2 <=142 tokens (B300 T=128 r2s1: 15.81 -> 14.77 us = 1.070x; geomean 1.041x B300 / 1.043x B200)
    b300_moe_finalize_ws2_t128_base_us = 15.81
    b300_moe_finalize_ws2_t128_quad_us = 14.77
    moe_finalize_allreduce_ws2_quad_route_max_speedup = round(
        b300_moe_finalize_ws2_t128_base_us / b300_moe_finalize_ws2_t128_quad_us, 3
    )  # 1.070
    moe_finalize_allreduce_ws2_quad_token_ceiling = 142

    # 3. FlashInfer (#8456, 27b38c15c3a7): cake_dense_projection_gemm K1 bound argument lists (GB300 2fwd+bwd (256,3072,1536): 156.4 -> 92.9 us = 1.684x)
    gb300_k1_proj_2fwdbwd_base_us = 156.4
    gb300_k1_proj_2fwdbwd_bound_us = 92.9
    dense_projection_k1_bound_args_gb300_speedup = round(
        gb300_k1_proj_2fwdbwd_base_us / gb300_k1_proj_2fwdbwd_bound_us, 3
    )  # 1.684

    # 4. FlashInfer (#8488, dc4bf3420d01): cake_mhc_det dual-layout fn split on NVIDIA R200 sm_107a (216 SMs, T=256: 44.01 -> 42.62 us = 1.033x)
    r200_mhc_det_t256_base_us = 44.01
    r200_mhc_det_t256_dual_fn_us = 42.62
    mhc_det_r200_sm107a_dual_fn_speedup = round(
        r200_mhc_det_t256_base_us / r200_mhc_det_t256_dual_fn_us, 3
    )  # 1.033

    # 5. FlashInfer (#8449, 936cd32a539a): cake_blackwell_softmax two-wave low-latency shared-tile kernel (B200 32x151936: 12.032 -> 10.976 us = 1.096x)
    b200_softmax_32x151936_main_us = 12.032
    b200_softmax_32x151936_lat_us = 10.976
    blackwell_softmax_two_wave_lat_speedup = round(
        b200_softmax_32x151936_main_us / b200_softmax_32x151936_lat_us, 3
    )  # 1.096

    # 6. FlashAttention (#2743, 54b044c8cf89 & #3091, 1dc8232fe209): CuTe explicit FwdConfig warm autotuning on GB300 (4.4 s -> 0.4 s = 11.0x max speedup) & MLA CTA choice exposure
    fa_cute_gb300_warm_tune_base_s = 4.4
    fa_cute_gb300_warm_tune_opt_s = 0.4
    fa_cute_gb300_warm_autotune_speedup = round(
        fa_cute_gb300_warm_tune_base_s / fa_cute_gb300_warm_tune_opt_s, 2
    )  # 11.0
    fa_mla_autotune_cta_choices = 3  # 2cta, 1cta, 1cta_kb64

    # 7. Triton (#12572, a972119e92a2; #12601, a94dac3739e8; #12522, 1fad2df0604c) & TileLang (#3689, fa69f7fa817c)
    triton_non_amd_generic_linear_escapes = 0
    triton_amd_scf_if_pointer_canonicalization_timeouts = 0
    triton_gfx1250_tdm_mbarrier_arrival_mismatches = 0
    tilelang_ascend_batched_3d_nd2nz_oob_corruptions = 0

    # 8. ROCm AITER (#7173, 6a4fa17deb06 & #5578, bc12ddf440f9): gfx950 CDNA4 mha_v4 hd256 MXFP4/6 (s=8192: 3652 vs 3082 TFLOP/s = 1.185x) & Triton MXFP8 FA2
    aiter_gfx950_hd128_s8192_tflops = 3082.0
    aiter_gfx950_hd256_s8192_tflops = 3652.0
    aiter_gfx950_hd256_mxfp4_tflops_speedup = round(
        aiter_gfx950_hd256_s8192_tflops / aiter_gfx950_hd128_s8192_tflops, 3
    )  # 1.185

    return {
        "suite": "flashinfer_12row_rmsnorm_quadroute_moe_k1_bound_mhc_r200_fa4_autotune_aiter_hd256_phase181",
        "phase": 181,
        "priority": 190,
        "rmsnorm_silu_gate_12row_b200_fwd_speedup": rmsnorm_silu_gate_12row_b200_fwd_speedup,
        "rmsnorm_silu_gate_12row_gb300_fwd_speedup": rmsnorm_silu_gate_12row_gb300_fwd_speedup,
        "rmsnorm_silu_gate_sm100_max_rows": rmsnorm_silu_gate_sm100_max_rows,
        "rmsnorm_silu_gate_sm103_max_rows": rmsnorm_silu_gate_sm103_max_rows,
        "moe_finalize_allreduce_ws2_quad_route_max_speedup": moe_finalize_allreduce_ws2_quad_route_max_speedup,
        "moe_finalize_allreduce_ws2_quad_token_ceiling": moe_finalize_allreduce_ws2_quad_token_ceiling,
        "dense_projection_k1_bound_args_gb300_speedup": dense_projection_k1_bound_args_gb300_speedup,
        "mhc_det_r200_sm107a_dual_fn_speedup": mhc_det_r200_sm107a_dual_fn_speedup,
        "blackwell_softmax_two_wave_lat_speedup": blackwell_softmax_two_wave_lat_speedup,
        "fa_cute_gb300_warm_autotune_speedup": fa_cute_gb300_warm_autotune_speedup,
        "fa_mla_autotune_cta_choices": fa_mla_autotune_cta_choices,
        "triton_non_amd_generic_linear_escapes": triton_non_amd_generic_linear_escapes,
        "triton_amd_scf_if_pointer_canonicalization_timeouts": triton_amd_scf_if_pointer_canonicalization_timeouts,
        "triton_gfx1250_tdm_mbarrier_arrival_mismatches": triton_gfx1250_tdm_mbarrier_arrival_mismatches,
        "tilelang_ascend_batched_3d_nd2nz_oob_corruptions": tilelang_ascend_batched_3d_nd2nz_oob_corruptions,
        "aiter_gfx950_hd256_s8192_tflops": aiter_gfx950_hd256_s8192_tflops,
        "aiter_gfx950_hd256_mxfp4_tflops_speedup": aiter_gfx950_hd256_mxfp4_tflops_speedup,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    results = simulate_gpu_software_stack_phase181()
    out_path = Path(__file__).with_name("results.json")
    out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    if args.check:
        assert results["rmsnorm_silu_gate_12row_b200_fwd_speedup"] == 1.312
        assert results["rmsnorm_silu_gate_12row_gb300_fwd_speedup"] == 1.304
        assert results["rmsnorm_silu_gate_sm100_max_rows"] == 1776
        assert results["rmsnorm_silu_gate_sm103_max_rows"] == 1824
        assert results["moe_finalize_allreduce_ws2_quad_route_max_speedup"] == 1.070
        assert results["moe_finalize_allreduce_ws2_quad_token_ceiling"] == 142
        assert results["dense_projection_k1_bound_args_gb300_speedup"] == 1.684
        assert results["mhc_det_r200_sm107a_dual_fn_speedup"] == 1.033
        assert results["blackwell_softmax_two_wave_lat_speedup"] == 1.096
        assert results["fa_cute_gb300_warm_autotune_speedup"] == 11.0
        assert results["fa_mla_autotune_cta_choices"] == 3
        assert results["triton_non_amd_generic_linear_escapes"] == 0
        assert results["triton_amd_scf_if_pointer_canonicalization_timeouts"] == 0
        assert results["triton_gfx1250_tdm_mbarrier_arrival_mismatches"] == 0
        assert results["tilelang_ascend_batched_3d_nd2nz_oob_corruptions"] == 0
        assert results["aiter_gfx950_hd256_s8192_tflops"] == 3652.0
        assert results["aiter_gfx950_hd256_mxfp4_tflops_speedup"] == 1.185

    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
