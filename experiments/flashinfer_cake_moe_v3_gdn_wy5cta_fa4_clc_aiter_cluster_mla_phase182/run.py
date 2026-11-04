#!/usr/bin/env python3
"""Phase 182: FlashInfer Cake Mega-MoE v3 Single-FC2-Task SM103a TP8/EP1, 5-CTA/SM Packed-Triangular GDN WY & SM107a K=64 DSV4 Decode, FlashAttention-4 Dense 2CTA MLA Head Padding & SM100 Causal CLC Defaults, Triton Decoded NVFP4 SMEM Accounting & gfx1250 LDS Prefetch, TileLang SM103a/SM107a TMA Gates, and ROCm AITER gfx1250 Cluster Split-K MLA & gfx950 Q256 MXFP4 Prefill."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def simulate_gpu_software_stack_phase182() -> dict[str, object]:
    # 1. FlashInfer (#8669, 1ce8f915659e): cake_mega_moe_v3 TP8/EP1 T32/T128/T512/T1024 on GB300 sm_103a (1 FC2 task/group covering all 4 N-clusters)
    # T32: Cake prev 91.26 -> 90.36 us (1.010x); vs trtllm_fp4_block_scale_routed_moe: T32 105.88 -> 90.27 us (1.173x), TP8/EP8 T4096 209.05 -> 178.14 us (1.174x)
    gb300_cake_moe_t32_prev_us = 91.26
    gb300_cake_moe_t32_new_us = 90.36
    cake_mega_moe_v3_t32_self_speedup = round(gb300_cake_moe_t32_prev_us / gb300_cake_moe_t32_new_us, 3)  # 1.010
    trtllm_fp4_routed_moe_ep8_t4096_us = 209.05
    cake_mega_moe_v3_ep8_t4096_us = 178.14
    cake_mega_moe_v3_vs_trtllm_max_speedup = round(
        trtllm_fp4_routed_moe_ep8_t4096_us / cake_mega_moe_v3_ep8_t4096_us, 3
    )  # 1.174
    cake_mega_moe_v3_fc2_tasks_per_group = 1

    # 2. FlashInfer (#8665, 8e7be5e3a832 & #8638, 283abf0d4eab): cake_gdn 5-CTA/SM wy_tc2 packed-triangular WY program on B200/GB300 & KCP checkpoint tape handoff
    # GB300 1/3 shape fwd+bwd: cuDNN FROST chunk-64 1.2787 ms vs Cake 0.6351 ms (2.013x); R200 1/3 fwd+bwd: 0.6136 ms vs 0.2428 ms (2.527x)
    gb300_gdn_frost_fwdbwd_ms = 1.2787
    gb300_gdn_cake_5cta_fwdbwd_ms = 0.6351
    cake_gdn_5cta_gb300_vs_frost_speedup = round(
        gb300_gdn_frost_fwdbwd_ms / gb300_gdn_cake_5cta_fwdbwd_ms, 3
    )  # 2.013
    r200_gdn_frost_fwdbwd_ms = 0.6136
    r200_gdn_cake_fwdbwd_ms = 0.2428
    cake_gdn_r200_vs_frost_max_speedup = round(
        r200_gdn_frost_fwdbwd_ms / r200_gdn_cake_fwdbwd_ms, 3
    )  # 2.527
    # #8638 KCP graph-replay checkpoint tape handoff at 32K tokens on R200: 3.554 -> 2.828 ms (1.257x)
    r200_gdn_kcp_ckpt_prev_ms = 3.554
    r200_gdn_kcp_ckpt_tape_handoff_ms = 2.828
    cake_gdn_kcp_tape_handoff_r200_speedup = round(
        r200_gdn_kcp_ckpt_prev_ms / r200_gdn_kcp_ckpt_tape_handoff_ms, 3
    )  # 1.257

    # 3. FlashInfer (#8668, a8e34e486334 & #8650, 02fa872444d6): cake_dsv4 SM107a 128-head uniform-gather K=64 FP8 MMA (rows >= 48) & bound-launch dense projection GEMM
    vr200_dsv4_h128_r124_k32_us = 14.623
    vr200_dsv4_h128_r124_k64_us = 11.444
    cake_dsv4_sm107a_k64_max_speedup = round(
        vr200_dsv4_h128_r124_k32_us / vr200_dsv4_h128_r124_k64_us, 4
    )  # 1.2778
    cake_dsv4_sm107a_k64_min_rows = 48
    gb200_proj_gemm_bwd_prev_us = 56.58
    gb200_proj_gemm_bwd_bound_us = 50.10
    cake_proj_gemm_bound_launch_gb200_speedup = round(
        gb200_proj_gemm_bwd_prev_us / gb200_proj_gemm_bwd_bound_us, 3
    )  # 1.129

    # 4. FlashAttention (#3118, 5f7b05a78bf0 & #3062, 6d6392ddc66d): Dense 2CTA absorbed-MLA head padding (65-127 -> 128) & SM100 causal CLC + Global LPT defaults
    b200_mla_96h_1cta_b512_sk256_us = 152.4
    b200_mla_96h_2cta_padded_b512_sk256_us = 68.5
    fa_mla_2cta_dense_96h_decode_speedup = round(
        b200_mla_96h_1cta_b512_sk256_us / b200_mla_96h_2cta_padded_b512_sk256_us, 3
    )  # 2.225
    fa_sm100_clc_global_lpt_gb300_max_speedup = 1.41
    fa_sm100_clc_global_lpt_max_tiles_per_sm = 2.25

    # 5. Triton (#12613, 90e50ecd9cb4 & #12524, 51f4f818afc1) & TileLang (#3772, 8587866b22c1): Decoded NVFP4 SMEM accounting on GB300 & TMA arch gates
    triton_gb300_fp8xnvfp4_constrained_us = 2711.89
    triton_gb300_fp8xnvfp4_new_default_us = 2057.72
    triton_gb300_fp8xnvfp4_speedup = round(
        triton_gb300_fp8xnvfp4_constrained_us / triton_gb300_fp8xnvfp4_new_default_us, 3
    )  # 1.318
    triton_gb300_smem_limit_bytes = 232448
    tilelang_cutedsl_new_tma_archs = 2  # sm_103a, sm_107a

    # 6. ROCm AITER (#7390, 0f61f2b95d0b & #5988, 0e072ca50ee7): gfx1250 MLA v4.1 in-kernel cluster split-K & gfx950 Q256 fused MXFP4 prefill
    aiter_gfx1250_mla_h64_n8_topk1024_before_us = 15.26
    aiter_gfx1250_mla_h64_n8_topk1024_cluster_us = 8.36
    aiter_gfx1250_mla_cluster_splitk_speedup = round(
        aiter_gfx1250_mla_h64_n8_topk1024_before_us / aiter_gfx1250_mla_h64_n8_topk1024_cluster_us, 3
    )  # 1.825
    aiter_gfx950_q256_mxfp4_vs_flydsl_max_gain_pct = 8.61

    return {
        "suite": "flashinfer_cake_moe_v3_gdn_wy5cta_fa4_clc_aiter_cluster_mla_phase182",
        "phase": 182,
        "priority": 191,
        "cake_mega_moe_v3_t32_self_speedup": cake_mega_moe_v3_t32_self_speedup,
        "cake_mega_moe_v3_vs_trtllm_max_speedup": cake_mega_moe_v3_vs_trtllm_max_speedup,
        "cake_mega_moe_v3_fc2_tasks_per_group": cake_mega_moe_v3_fc2_tasks_per_group,
        "cake_gdn_5cta_gb300_vs_frost_speedup": cake_gdn_5cta_gb300_vs_frost_speedup,
        "cake_gdn_r200_vs_frost_max_speedup": cake_gdn_r200_vs_frost_max_speedup,
        "cake_gdn_kcp_tape_handoff_r200_speedup": cake_gdn_kcp_tape_handoff_r200_speedup,
        "cake_dsv4_sm107a_k64_max_speedup": cake_dsv4_sm107a_k64_max_speedup,
        "cake_dsv4_sm107a_k64_min_rows": cake_dsv4_sm107a_k64_min_rows,
        "cake_proj_gemm_bound_launch_gb200_speedup": cake_proj_gemm_bound_launch_gb200_speedup,
        "fa_mla_2cta_dense_96h_decode_speedup": fa_mla_2cta_dense_96h_decode_speedup,
        "fa_sm100_clc_global_lpt_gb300_max_speedup": fa_sm100_clc_global_lpt_gb300_max_speedup,
        "fa_sm100_clc_global_lpt_max_tiles_per_sm": fa_sm100_clc_global_lpt_max_tiles_per_sm,
        "triton_gb300_fp8xnvfp4_speedup": triton_gb300_fp8xnvfp4_speedup,
        "triton_gb300_smem_limit_bytes": triton_gb300_smem_limit_bytes,
        "tilelang_cutedsl_new_tma_archs": tilelang_cutedsl_new_tma_archs,
        "aiter_gfx1250_mla_cluster_splitk_speedup": aiter_gfx1250_mla_cluster_splitk_speedup,
        "aiter_gfx950_q256_mxfp4_vs_flydsl_max_gain_pct": aiter_gfx950_q256_mxfp4_vs_flydsl_max_gain_pct,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    results = simulate_gpu_software_stack_phase182()
    out_path = Path(__file__).with_name("results.json")
    out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    if args.check:
        assert results["cake_mega_moe_v3_t32_self_speedup"] == 1.010
        assert results["cake_mega_moe_v3_vs_trtllm_max_speedup"] == 1.174
        assert results["cake_mega_moe_v3_fc2_tasks_per_group"] == 1
        assert results["cake_gdn_5cta_gb300_vs_frost_speedup"] == 2.013
        assert results["cake_gdn_r200_vs_frost_max_speedup"] == 2.527
        assert results["cake_gdn_kcp_tape_handoff_r200_speedup"] == 1.257
        assert results["cake_dsv4_sm107a_k64_max_speedup"] == 1.2778
        assert results["cake_dsv4_sm107a_k64_min_rows"] == 48
        assert results["cake_proj_gemm_bound_launch_gb200_speedup"] == 1.129
        assert results["fa_mla_2cta_dense_96h_decode_speedup"] == 2.225
        assert results["fa_sm100_clc_global_lpt_gb300_max_speedup"] == 1.41
        assert results["fa_sm100_clc_global_lpt_max_tiles_per_sm"] == 2.25
        assert results["triton_gb300_fp8xnvfp4_speedup"] == 1.318
        assert results["triton_gb300_smem_limit_bytes"] == 232448
        assert results["tilelang_cutedsl_new_tma_archs"] == 2
        assert results["aiter_gfx1250_mla_cluster_splitk_speedup"] == 1.825
        assert results["aiter_gfx950_q256_mxfp4_vs_flydsl_max_gain_pct"] == 8.61

    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
