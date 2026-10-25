from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def build_summary() -> dict[str, Any]:
    # 1. cake_gated_act_mxfp8 (PR #7346, 0b352a63)
    mxfp8_small_tile = (32, 32)
    mxfp8_sm103_colwise_band_tile = (32, 64)
    mxfp8_sm103_colwise_band_min_elems = 1 << 22  # 4,194,304
    mxfp8_sm103_colwise_band_max_elems = 1 << 24  # 16,777,216
    mxfp8_large_rows_per_thread = 4

    b200_128x128_fwd_colwise_prev_us = 2.67
    b200_128x128_fwd_colwise_new_us = 2.12
    b200_128x128_fwd_colwise_speedup = round(
        b200_128x128_fwd_colwise_prev_us / b200_128x128_fwd_colwise_new_us, 3
    )

    gb300_128x128_fwd_colwise_prev_us = 2.26
    gb300_128x128_fwd_colwise_new_us = 2.02
    gb300_128x128_fwd_colwise_speedup = round(
        gb300_128x128_fwd_colwise_prev_us / gb300_128x128_fwd_colwise_new_us, 3
    )

    gb300_131072x8192_fwd_both_prev_us = 923.27
    gb300_131072x8192_fwd_both_new_us = 904.89
    gb300_131072x8192_fwd_both_speedup = round(
        gb300_131072x8192_fwd_both_prev_us / gb300_131072x8192_fwd_both_new_us, 3
    )

    r200_128x128_bwd_colwise_prev_us = 2.42
    r200_128x128_bwd_colwise_new_us = 2.17
    r200_128x128_bwd_colwise_speedup = round(
        r200_128x128_bwd_colwise_prev_us / r200_128x128_bwd_colwise_new_us, 3
    )

    r200_131072x8192_fwd_rowwise_prev_us = 528.30
    r200_131072x8192_fwd_rowwise_new_us = 481.99
    r200_131072x8192_fwd_rowwise_speedup = round(
        r200_131072x8192_fwd_rowwise_prev_us / r200_131072x8192_fwd_rowwise_new_us, 3
    )

    # 2. cake_all_gather_matmul (PR #7360, e7f5b1e1)
    ag_m_per_rank = 1025
    ag_m_padded_per_rank = 1152
    ag_n = 1280
    ag_k = 8192
    ag_world_size = 8
    ag_threads = 192
    ag_dynamic_smem_bytes = 197632
    ag_tmem_ncols = 512
    ag_final_pass_base_tiles_p = (ag_m_padded_per_rank // 128) * (ag_n // 256)  # 9 * 5 = 45
    ag_final_pass_nsplit_tiles_2p = 2 * ag_final_pass_base_tiles_p  # 90 <= 148 CTAs
    ag_node1_prev_us = 341.6
    ag_node1_new_us = 336.6
    ag_node1_speedup = 1.0116
    ag_node2_prev_us = 340.7
    ag_node2_new_us = 337.7
    ag_node2_speedup = 1.0209

    # 3. cake_mm_fp4 per-token NVFP4 quantizer + GEMM (PR #7356, 941fda25)
    nvfp4_row_split_cluster_ctas = 4
    nvfp4_splitk3_cluster = 3
    nvfp4_b200_quant_k28672_m17_prev_us = 3.601
    nvfp4_b200_quant_k28672_m17_new_us = 2.874
    nvfp4_b200_quant_max_speedup = 1.255
    nvfp4_gb300_quant_k28672_m1_prev_us = 3.235
    nvfp4_gb300_quant_k28672_m1_new_us = 2.604
    nvfp4_gb300_quant_max_speedup = 1.243
    nvfp4_gb300_gemm_k8192_n28672_m8192_prev_us = 519.5
    nvfp4_gb300_gemm_k8192_n28672_m8192_new_us = 454.5
    nvfp4_gb300_gemm_max_speedup = 1.144
    nvfp4_gb300_fused_k8192_n28672_m8192_prev_us = 540.6
    nvfp4_gb300_fused_k8192_n28672_m8192_new_us = 474.0
    nvfp4_gb300_fused_max_speedup = 1.140

    # At M=64, N=2048, K=7168: 8-token tile (M_tiles=8, N_tiles=8, splitK=3) -> 64 clusters * 3 CTAs = 192 CTAs > 152 SMs
    # 16-token tile (M_tiles=4, N_tiles=8, splitK=3) -> 32 clusters * 3 CTAs = 96 CTAs <= 148/152 SMs
    nvfp4_m8_splitk3_ctas_at_m64_n2048 = (64 // 8) * (2048 // 256) * nvfp4_splitk3_cluster
    nvfp4_m16_splitk3_ctas_at_m64_n2048 = (64 // 16) * (2048 // 256) * nvfp4_splitk3_cluster

    # 4. cake_kimi_k3_fused_router (PR #7355, ee50fd01)
    kimi_router_num_experts = 896
    kimi_router_top_k = 16
    kimi_router_cells = 28
    kimi_gb300_t16_bm8_prev_us = 5.89
    kimi_gb300_t16_bm8_new_us = 3.39
    kimi_gb300_t16_bm8_speedup = 1.74
    kimi_gb300_t8192_bm16_prev_us = 40.66
    kimi_gb300_t8192_bm16_new_us = 22.61
    kimi_gb300_t8192_bm16_speedup = 1.80
    kimi_b200_t16_bm8_prev_us = 6.13
    kimi_b200_t16_bm8_new_us = 3.54
    kimi_b200_t16_bm8_speedup = 1.73
    kimi_b200_t8192_bm16_prev_us = 40.64
    kimi_b200_t8192_bm16_new_us = 23.94
    kimi_b200_t8192_bm16_speedup = 1.70

    # 5. cake_qsa_nvfp4 SM107 persistent walk (PR #7353, 1cfd7312)
    qsa_sm107_sms = 212
    qsa_sm107_desc_mask = 0x7FFF  # 15-bit SMEM descriptor field on CC 10.7
    qsa_sm100_desc_mask = 0x3FFF  # 14-bit SMEM descriptor field on CC 10.0/10.3
    qsa_rows = 128
    qsa_kv_heads = 2
    qsa_plain_splits = 2
    qsa_work_items = qsa_rows * qsa_kv_heads * qsa_plain_splits  # 512
    qsa_waves_on_r200 = math.ceil(qsa_work_items / qsa_sm107_sms)  # ceil(512/212) = 3 <= 3
    qsa_partial_slots_per_pair = 2
    qsa_r200_nhd_prev_us = 54.99
    qsa_r200_nhd_new_us = 47.15
    qsa_r200_nhd_speedup = 1.17
    qsa_r200_hnd_prev_us = 55.71
    qsa_r200_hnd_new_us = 46.05
    qsa_r200_hnd_speedup = 1.21

    # 6. cake_warp_decode 5-deep-ring K256 FC2 & vector finalize (PR #7357, ea95347a)
    warp_decode_prev_ring_depth = 3
    warp_decode_new_ring_depth = 5
    warp_decode_tile_meta_rows = 512
    warp_decode_tile_meta_bytes = 8192
    warp_decode_prev_smem_bytes = 179456
    warp_decode_new_smem_bytes = 187648
    warp_decode_finalize_features_per_thread = 4
    warp_decode_b200_m4_read_prev_us = 23.58
    warp_decode_b200_m4_read_new_us = 21.19
    warp_decode_b200_m4_speedup = round(
        warp_decode_b200_m4_read_prev_us / warp_decode_b200_m4_read_new_us, 3
    )
    warp_decode_gb300_m4_read_prev_us = 22.87
    warp_decode_gb300_m4_read_new_us = 20.72
    warp_decode_gb300_m4_speedup = round(
        warp_decode_gb300_m4_read_prev_us / warp_decode_gb300_m4_read_new_us, 3
    )

    # 7. cake_fused_norm_combine (PR #7352, 20ede145) & cake_kimi_k3_attn_res (PR #7338, e9261a5e)
    norm_combine_prev_programs = 4
    norm_combine_new_programs = 6
    norm_combine_3cta_band = (593, 1772)
    norm_combine_b200_t1_prev_us = 8.736
    norm_combine_b200_t1_new_us = 8.032
    norm_combine_b200_t1_speedup = 1.088
    norm_combine_b300_t1_prev_us = 9.153
    norm_combine_b300_t1_new_us = 8.384
    norm_combine_b300_t1_speedup = 1.092

    attn_res_prev_programs = 138
    attn_res_new_programs = 181
    attn_res_prev_libraries = 167
    attn_res_new_libraries = 211
    attn_res_gb300_m192_k3_prev_us = 8.416
    attn_res_gb300_m192_k3_new_us = 7.584
    attn_res_gb300_max_speedup = 1.1097
    attn_res_b300_m192_k3_prev_us = 8.736
    attn_res_b300_m192_k3_new_us = 7.776
    attn_res_b300_max_speedup = 1.1219
    attn_res_b200_m192_k2_pdl_prev_us = 8.000
    attn_res_b200_m192_k2_pdl_new_us = 7.072
    attn_res_b200_max_speedup = 1.1297

    return {
        "experiment_id": "flashinfer_sm10x_mxfp8_nvfp4_kimi_qsa_ag_norm_phase85",
        "phase": 85,
        "priority": 94,
        "cake_gated_act_mxfp8": {
            "small_tile": list(mxfp8_small_tile),
            "sm103_colwise_band_tile": list(mxfp8_sm103_colwise_band_tile),
            "sm103_colwise_band_elems": [
                mxfp8_sm103_colwise_band_min_elems,
                mxfp8_sm103_colwise_band_max_elems,
            ],
            "large_rows_per_thread": mxfp8_large_rows_per_thread,
            "b200_128x128_fwd_colwise_prev_us": b200_128x128_fwd_colwise_prev_us,
            "b200_128x128_fwd_colwise_new_us": b200_128x128_fwd_colwise_new_us,
            "b200_128x128_fwd_colwise_speedup": b200_128x128_fwd_colwise_speedup,
            "gb300_128x128_fwd_colwise_prev_us": gb300_128x128_fwd_colwise_prev_us,
            "gb300_128x128_fwd_colwise_new_us": gb300_128x128_fwd_colwise_new_us,
            "gb300_128x128_fwd_colwise_speedup": gb300_128x128_fwd_colwise_speedup,
            "gb300_131072x8192_fwd_both_prev_us": gb300_131072x8192_fwd_both_prev_us,
            "gb300_131072x8192_fwd_both_new_us": gb300_131072x8192_fwd_both_new_us,
            "gb300_131072x8192_fwd_both_speedup": gb300_131072x8192_fwd_both_speedup,
            "r200_128x128_bwd_colwise_prev_us": r200_128x128_bwd_colwise_prev_us,
            "r200_128x128_bwd_colwise_new_us": r200_128x128_bwd_colwise_new_us,
            "r200_128x128_bwd_colwise_speedup": r200_128x128_bwd_colwise_speedup,
            "r200_131072x8192_fwd_rowwise_prev_us": r200_131072x8192_fwd_rowwise_prev_us,
            "r200_131072x8192_fwd_rowwise_new_us": r200_131072x8192_fwd_rowwise_new_us,
            "r200_131072x8192_fwd_rowwise_speedup": r200_131072x8192_fwd_rowwise_speedup,
        },
        "cake_all_gather_matmul": {
            "m_per_rank": ag_m_per_rank,
            "m_padded_per_rank": ag_m_padded_per_rank,
            "n": ag_n,
            "k": ag_k,
            "world_size": ag_world_size,
            "threads": ag_threads,
            "dynamic_smem_bytes": ag_dynamic_smem_bytes,
            "tmem_ncols": ag_tmem_ncols,
            "final_pass_base_tiles_p": ag_final_pass_base_tiles_p,
            "final_pass_nsplit_tiles_2p": ag_final_pass_nsplit_tiles_2p,
            "node1_prev_us": ag_node1_prev_us,
            "node1_new_us": ag_node1_new_us,
            "node1_speedup": ag_node1_speedup,
            "node2_prev_us": ag_node2_prev_us,
            "node2_new_us": ag_node2_new_us,
            "node2_speedup": ag_node2_speedup,
        },
        "cake_mm_fp4": {
            "row_split_cluster_ctas": nvfp4_row_split_cluster_ctas,
            "splitk3_cluster": nvfp4_splitk3_cluster,
            "b200_quant_k28672_m17_prev_us": nvfp4_b200_quant_k28672_m17_prev_us,
            "b200_quant_k28672_m17_new_us": nvfp4_b200_quant_k28672_m17_new_us,
            "b200_quant_max_speedup": nvfp4_b200_quant_max_speedup,
            "gb300_quant_k28672_m1_prev_us": nvfp4_gb300_quant_k28672_m1_prev_us,
            "gb300_quant_k28672_m1_new_us": nvfp4_gb300_quant_k28672_m1_new_us,
            "gb300_quant_max_speedup": nvfp4_gb300_quant_max_speedup,
            "gb300_gemm_k8192_n28672_m8192_prev_us": nvfp4_gb300_gemm_k8192_n28672_m8192_prev_us,
            "gb300_gemm_k8192_n28672_m8192_new_us": nvfp4_gb300_gemm_k8192_n28672_m8192_new_us,
            "gb300_gemm_max_speedup": nvfp4_gb300_gemm_max_speedup,
            "gb300_fused_k8192_n28672_m8192_prev_us": nvfp4_gb300_fused_k8192_n28672_m8192_prev_us,
            "gb300_fused_k8192_n28672_m8192_new_us": nvfp4_gb300_fused_k8192_n28672_m8192_new_us,
            "gb300_fused_max_speedup": nvfp4_gb300_fused_max_speedup,
            "m8_splitk3_ctas_at_m64_n2048": nvfp4_m8_splitk3_ctas_at_m64_n2048,
            "m16_splitk3_ctas_at_m64_n2048": nvfp4_m16_splitk3_ctas_at_m64_n2048,
        },
        "cake_kimi_k3_fused_router": {
            "num_experts": kimi_router_num_experts,
            "top_k": kimi_router_top_k,
            "cells": kimi_router_cells,
            "gb300_t16_bm8_prev_us": kimi_gb300_t16_bm8_prev_us,
            "gb300_t16_bm8_new_us": kimi_gb300_t16_bm8_new_us,
            "gb300_t16_bm8_speedup": kimi_gb300_t16_bm8_speedup,
            "gb300_t8192_bm16_prev_us": kimi_gb300_t8192_bm16_prev_us,
            "gb300_t8192_bm16_new_us": kimi_gb300_t8192_bm16_new_us,
            "gb300_t8192_bm16_speedup": kimi_gb300_t8192_bm16_speedup,
            "b200_t16_bm8_prev_us": kimi_b200_t16_bm8_prev_us,
            "b200_t16_bm8_new_us": kimi_b200_t16_bm8_new_us,
            "b200_t16_bm8_speedup": kimi_b200_t16_bm8_speedup,
            "b200_t8192_bm16_prev_us": kimi_b200_t8192_bm16_prev_us,
            "b200_t8192_bm16_new_us": kimi_b200_t8192_bm16_new_us,
            "b200_t8192_bm16_speedup": kimi_b200_t8192_bm16_speedup,
        },
        "cake_qsa_nvfp4_sm107": {
            "sm107_sms": qsa_sm107_sms,
            "sm107_desc_mask": qsa_sm107_desc_mask,
            "sm100_desc_mask": qsa_sm100_desc_mask,
            "rows": qsa_rows,
            "kv_heads": qsa_kv_heads,
            "plain_splits": qsa_plain_splits,
            "work_items": qsa_work_items,
            "waves_on_r200": qsa_waves_on_r200,
            "partial_slots_per_pair": qsa_partial_slots_per_pair,
            "r200_nhd_prev_us": qsa_r200_nhd_prev_us,
            "r200_nhd_new_us": qsa_r200_nhd_new_us,
            "r200_nhd_speedup": qsa_r200_nhd_speedup,
            "r200_hnd_prev_us": qsa_r200_hnd_prev_us,
            "r200_hnd_new_us": qsa_r200_hnd_new_us,
            "r200_hnd_speedup": qsa_r200_hnd_speedup,
        },
        "cake_warp_decode": {
            "prev_ring_depth": warp_decode_prev_ring_depth,
            "new_ring_depth": warp_decode_new_ring_depth,
            "tile_meta_rows": warp_decode_tile_meta_rows,
            "tile_meta_bytes": warp_decode_tile_meta_bytes,
            "prev_smem_bytes": warp_decode_prev_smem_bytes,
            "new_smem_bytes": warp_decode_new_smem_bytes,
            "finalize_features_per_thread": warp_decode_finalize_features_per_thread,
            "b200_m4_read_prev_us": warp_decode_b200_m4_read_prev_us,
            "b200_m4_read_new_us": warp_decode_b200_m4_read_new_us,
            "b200_m4_speedup": warp_decode_b200_m4_speedup,
            "gb300_m4_read_prev_us": warp_decode_gb300_m4_read_prev_us,
            "gb300_m4_read_new_us": warp_decode_gb300_m4_read_new_us,
            "gb300_m4_speedup": warp_decode_gb300_m4_speedup,
        },
        "cake_norm_combine_and_attn_res": {
            "norm_combine_prev_programs": norm_combine_prev_programs,
            "norm_combine_new_programs": norm_combine_new_programs,
            "norm_combine_3cta_band": list(norm_combine_3cta_band),
            "norm_combine_b200_t1_prev_us": norm_combine_b200_t1_prev_us,
            "norm_combine_b200_t1_new_us": norm_combine_b200_t1_new_us,
            "norm_combine_b200_t1_speedup": norm_combine_b200_t1_speedup,
            "norm_combine_b300_t1_prev_us": norm_combine_b300_t1_prev_us,
            "norm_combine_b300_t1_new_us": norm_combine_b300_t1_new_us,
            "norm_combine_b300_t1_speedup": norm_combine_b300_t1_speedup,
            "attn_res_prev_programs": attn_res_prev_programs,
            "attn_res_new_programs": attn_res_new_programs,
            "attn_res_prev_libraries": attn_res_prev_libraries,
            "attn_res_new_libraries": attn_res_new_libraries,
            "attn_res_gb300_m192_k3_prev_us": attn_res_gb300_m192_k3_prev_us,
            "attn_res_gb300_m192_k3_new_us": attn_res_gb300_m192_k3_new_us,
            "attn_res_gb300_max_speedup": attn_res_gb300_max_speedup,
            "attn_res_b300_m192_k3_prev_us": attn_res_b300_m192_k3_prev_us,
            "attn_res_b300_m192_k3_new_us": attn_res_b300_m192_k3_new_us,
            "attn_res_b300_max_speedup": attn_res_b300_max_speedup,
            "attn_res_b200_m192_k2_pdl_prev_us": attn_res_b200_m192_k2_pdl_prev_us,
            "attn_res_b200_m192_k2_pdl_new_us": attn_res_b200_m192_k2_pdl_new_us,
            "attn_res_b200_max_speedup": attn_res_b200_max_speedup,
        },
    }


def validate_summary(summary: dict[str, Any]) -> None:
    mxfp8 = summary["cake_gated_act_mxfp8"]
    assert mxfp8["small_tile"] == [32, 32]
    assert mxfp8["sm103_colwise_band_tile"] == [32, 64]
    assert mxfp8["sm103_colwise_band_elems"] == [4194304, 16777216]
    assert mxfp8["b200_128x128_fwd_colwise_speedup"] == 1.259
    assert mxfp8["r200_131072x8192_fwd_rowwise_speedup"] == 1.096

    ag = summary["cake_all_gather_matmul"]
    assert ag["final_pass_base_tiles_p"] == 45
    assert ag["final_pass_nsplit_tiles_2p"] == 90
    assert ag["dynamic_smem_bytes"] == 197632

    nvfp4 = summary["cake_mm_fp4"]
    assert nvfp4["b200_quant_max_speedup"] == 1.255
    assert nvfp4["gb300_gemm_max_speedup"] == 1.144
    assert nvfp4["m8_splitk3_ctas_at_m64_n2048"] == 192
    assert nvfp4["m16_splitk3_ctas_at_m64_n2048"] == 96

    router = summary["cake_kimi_k3_fused_router"]
    assert router["cells"] == 28
    assert router["gb300_t8192_bm16_speedup"] == 1.80
    assert router["b200_t16_bm8_speedup"] == 1.73

    qsa = summary["cake_qsa_nvfp4_sm107"]
    assert qsa["sm107_desc_mask"] == 0x7FFF
    assert qsa["waves_on_r200"] == 3
    assert qsa["r200_hnd_speedup"] == 1.21

    wd = summary["cake_warp_decode"]
    assert wd["new_ring_depth"] == 5
    assert wd["new_smem_bytes"] - wd["prev_smem_bytes"] == 8192
    assert wd["b200_m4_speedup"] == 1.113

    nc = summary["cake_norm_combine_and_attn_res"]
    assert nc["norm_combine_new_programs"] == 6
    assert nc["attn_res_new_programs"] == 181
    assert nc["attn_res_new_libraries"] == 211


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    summary = build_summary()
    validate_summary(summary)

    out_path = Path(__file__).with_name("summary.json")
    if args.check and out_path.exists():
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        assert existing == summary, "summary.json drift detected"
    else:
        out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
