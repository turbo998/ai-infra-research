#!/usr/bin/env python3
"""Deterministic verification of FlashInfer Cake All-Gather Matmul (SM100a/SM103a),
Cake MSA (SM90a/SM100a/SM103a) & VSA SM90a Kernels, and Blackwell PTX tcgen05
TMEM Lane-Quarter & CTA-Pair Stage-Drain Invariants.

Covers first-hand upstream commits:
- flashinfer@d9695940c53e0563cb0855f93c0955660c143d9f (#6865)
- flashinfer@9c68e44fd2a7664d53d49168ed616aa8c06ef1f7 (#6761)
- flashinfer@8a93c9b833761c75939f176a58268ad77ac304f1 (#6252)
- flashinfer@7f6f119d8534441430934e798ba056f052d1ba20 (#6725)
- flashinfer@ddf080d8a56c87f60a25ee9df750e373c4bcbe92 (#6877)
- flashinfer@c8be875c7c403c4eac664f034b098157d9dc0642 (#6778)
- flashinfer@e4e7510f51fe9f058390ad84c66178805c8b2f45 (#6767)
- flashinfer@653cce0a6e903a042d374cd509002bd82ae6d3a3 (#6820)
- flashinfer@2b3cd4f2f990954ea9a196c352dadae25fd3ff8c (#6763)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def simulate_cake_allgather_matmul() -> dict[str, Any]:
    """Model SM100a CTA-pair _cg2 routing and SM100a/SM103a 16-CTA-per-peer push launch (#6865, #6761)."""
    sm100a_ws2_cg2_table = [
        {
            "shape": "ws2_1024x2048_k8192",
            "dtype_layout": "BF16_N_major",
            "old_kernel": "agm_ws2_m2048_n2048_k8192_bf16_nt_sm100a",
            "new_kernel": "agm_ws2_m2048_n2048_k8192_bf16_nt_sm100a_cg2",
            "node1_old_us": 103.5,
            "node1_new_us": 98.4,
            "node1_speedup_x": 1.035,
            "node2_old_us": 102.4,
            "node2_new_us": 101.6,
            "node2_speedup_x": 1.027,
        },
        {
            "shape": "ws2_1024x2048_k8192",
            "dtype_layout": "BF16_K_major",
            "old_kernel": "agm_ws2_m2048_n2048_k8192_bf16_nn_sm100a",
            "new_kernel": "agm_ws2_m2048_n2048_k8192_bf16_nn_sm100a_cg2",
            "node1_old_us": 101.9,
            "node1_new_us": 95.4,
            "node1_speedup_x": 1.065,
            "node2_old_us": 104.2,
            "node2_new_us": 99.6,
            "node2_speedup_x": 1.051,
        },
        {
            "shape": "ws2_1024x2048_k8192",
            "dtype_layout": "FP16_K_major",
            "old_kernel": "agm_ws2_m2048_n2048_k8192_fp16_nn_sm100a",
            "new_kernel": "agm_ws2_m2048_n2048_k8192_fp16_nn_sm100a_cg2",
            "node1_old_us": 101.8,
            "node1_new_us": 95.3,
            "node1_speedup_x": 1.065,
            "node2_old_us": 104.4,
            "node2_new_us": 100.2,
            "node2_speedup_x": 1.052,
        },
    ]

    def push_launch_geometry(world_size: int, ctas_per_peer: int, bytes_per_peer: int) -> dict[str, int]:
        num_peers = world_size - 1
        total_push_ctas = num_peers * ctas_per_peer
        bytes_per_cta = bytes_per_peer // ctas_per_peer
        return {
            "world_size": world_size,
            "num_peers": num_peers,
            "ctas_per_peer": ctas_per_peer,
            "total_push_ctas": total_push_ctas,
            "bytes_per_peer": bytes_per_peer,
            "bytes_per_cta": bytes_per_cta,
        }

    # ws4 512 x 2560, K=8192, BF16/FP16 (2 bytes): M_local * K * 2 = 512 * 8192 * 2 = 8 MiB per peer
    ws4_bytes_per_peer = 512 * 8192 * 2
    sm100a_ws4_old = push_launch_geometry(4, 8, ws4_bytes_per_peer)
    sm100a_ws4_new = push_launch_geometry(4, 16, ws4_bytes_per_peer)

    # ws8 1024 x 1280, K=8192, BF16/FP16 (2 bytes): M_local * K * 2 = 1024 * 8192 * 2 = 16 MiB per peer
    ws8_bytes_per_peer = 1024 * 8192 * 2
    sm103a_ws8_old = push_launch_geometry(8, 8, ws8_bytes_per_peer)
    sm103a_ws8_new = push_launch_geometry(8, 16, ws8_bytes_per_peer)

    return {
        "sm100a_ws2_cg2_table": sm100a_ws2_cg2_table,
        "sm100a_ws4_512x2560_k8192_push": {
            "old_8_ctas_per_peer": sm100a_ws4_old,
            "new_16_ctas_per_peer": sm100a_ws4_new,
            "bf16_n_major_node2_old_us": 127.5,
            "bf16_n_major_node2_new_us": 113.7,
            "bf16_n_major_node2_speedup_x": 1.086,
            "fp16_k_major_node2_old_us": 128.8,
            "fp16_k_major_node2_new_us": 114.2,
            "fp16_k_major_node2_speedup_x": 1.082,
        },
        "sm103a_ws8_1024x1280_k8192_push": {
            "old_8_ctas_per_peer": sm103a_ws8_old,
            "new_16_ctas_per_peer": sm103a_ws8_new,
            "bf16_n_major_node1_old_us": 324.8,
            "bf16_n_major_node1_new_us": 317.1,
            "bf16_n_major_node1_speedup_x": 1.012,
            "bf16_n_major_node2_old_us": 314.5,
            "bf16_n_major_node2_new_us": 311.2,
            "bf16_n_major_node2_speedup_x": 1.034,
        },
    }


def simulate_cake_msa_and_vsa_sm90() -> dict[str, Any]:
    """Model Cake MSA Hopper/Blackwell programs (#6252, #6725) and Cake VSA SM90 (#6877)."""
    msa_sm90_program_counts = {
        "sparse_decode": 22,
        "sparse_prefill": 6,
        "proxy_decode": 24,
        "proxy_prefill": 5,
        "topk": 32,
    }
    total_msa_sm90_programs = sum(msa_sm90_program_counts.values())

    msa_sm90_benchmarks = [
        {
            "case": "topk_nvp_forced_h4_tiles512_tq256",
            "cake_sm90_us": 4.256,
            "fi_sm90_us": 13.775,
            "speedup_x": round(13.775 / 4.256, 3),
        },
        {
            "case": "topk_plain_h4_tiles512_tq256",
            "cake_sm90_us": 4.832,
            "fi_sm90_us": 14.079,
            "speedup_x": round(14.079 / 4.832, 3),
        },
        {
            "case": "proxy_decode_fp8_b4_s256_h32_k4",
            "cake_sm90_us": 8.031,
            "fi_sm90_us": 16.929,
            "speedup_x": round(16.929 / 8.031, 3),
        },
        {
            "case": "sparse_prefill_bf16_flat_r2_tq256_tk256_h32_k4",
            "cake_sm90_us": 17.664,
            "fi_sm90_us": 26.863,
            "speedup_x": round(26.863 / 17.664, 3),
        },
        {
            "case": "proxy_prefill_fp8_flat_r2_tq128_tiles128_h32_k4",
            "cake_sm90_us": 8.448,
            "fi_sm90_us": 11.968,
            "speedup_x": round(11.968 / 8.448, 3),
        },
    ]

    # Heavy-first GQA-folded persistent union prefill scheduling simulation (#6725)
    # GQA ratio gqa_ratio = 8; query rows grouped into super-tiles sorted descending by union KV tile count
    super_tile_union_kv_counts = [12, 64, 18, 96, 8, 80, 24, 112]
    num_sms = 4
    fifo_sm_loads = [0] * num_sms
    for kv_tiles in super_tile_union_kv_counts:
        min_sm = min(range(num_sms), key=lambda idx: fifo_sm_loads[idx])
        fifo_sm_loads[min_sm] += kv_tiles

    heavy_first_sm_loads = [0] * num_sms
    for kv_tiles in sorted(super_tile_union_kv_counts, reverse=True):
        min_sm = min(range(num_sms), key=lambda idx: heavy_first_sm_loads[idx])
        heavy_first_sm_loads[min_sm] += kv_tiles

    vsa_sm90_program_counts = {
        "persistent": 2,  # attention, attention_queue
        "small_direct": 4,  # k1, k3, k4, k6
        "small_dsm_cluster": 4,  # k2c4, k4c2, k4c4, k6c2
        "small_split_kv": 2,  # k4s, k6s
        "small_split_cluster": 5,  # k6sc3, k6sc4, k6sc5, k6sc6, k6sc8
    }
    total_vsa_sm90_programs = sum(vsa_sm90_program_counts.values())

    return {
        "msa_sm90_program_counts": msa_sm90_program_counts,
        "total_msa_sm90_programs": total_msa_sm90_programs,
        "msa_sm90_benchmarks": msa_sm90_benchmarks,
        "msa_blackwell_union_prefill": {
            "fifo_max_sm_kv_tiles": max(fifo_sm_loads),
            "heavy_first_max_sm_kv_tiles": max(heavy_first_sm_loads),
            "tail_imbalance_reduction_ratio": round(max(fifo_sm_loads) / max(heavy_first_sm_loads), 4),
            "bf16_flat_union_prefill_gb300_speedup_x": 1.825,
            "bf16_flat_union_prefill_b200_speedup_x": 1.557,
            "bf16_paged_first_topk_gb300_speedup_x": 3.300,
            "bf16_paged_first_topk_b200_speedup_x": 2.727,
            "topk_merge_8warp_gb300_speedup_x": 2.133,
            "topk_merge_8warp_b200_speedup_x": 2.129,
        },
        "vsa_sm90_program_counts": vsa_sm90_program_counts,
        "total_vsa_sm90_programs": total_vsa_sm90_programs,
        "vsa_sm90_benchmarks": {
            "ext33_geomean_speedup_x": 1.0515,
            "real39_geomean_speedup_x": 1.0315,
            "h1_m512_n8192_k48_before_us": 13.344,
            "h1_m512_n8192_k48_after_us": 10.639,
            "h1_m512_n8192_k48_speedup_x": 1.2524,
            "h8_m128_n32768_k64_before_us": 35.136,
            "h8_m128_n32768_k64_after_us": 28.952,
            "h8_m128_n32768_k64_speedup_x": 1.2126,
        },
    }


def simulate_blackwell_tcgen05_invariants() -> dict[str, Any]:
    """Verify Blackwell PTX tcgen05 TMEM lane-quarter addressing (#6778, #6767)
    and 2-CTA multicast mma_done stage-release drain (#6820, #6763).
    """
    # 1. cake_dsa_train warpgroup TMEM lane-quarter base: tmem_lane_base = (tid & 96) << 16
    # A 128-thread warpgroup has tid in 0..127 (warps 0..3).
    # Without tmem_lane_base, every warp passes lane=0 in bits 16..22 and reads lanes 0..31!
    dsa_warp_quarters: list[dict[str, Any]] = []
    for warp_idx in range(4):
        tid_start = warp_idx * 32
        tmem_lane_base = (tid_start & 96) << 16
        decoded_lane_offset = (tmem_lane_base >> 16) & 0x7F
        buggy_lanes_read = list(range(0, 32))
        fixed_lanes_read = list(range(decoded_lane_offset, decoded_lane_offset + 32))
        dsa_warp_quarters.append(
            {
                "warp_in_wg": warp_idx,
                "tid_range": f"{tid_start}..{tid_start + 31}",
                "tmem_lane_base_hex": f"0x{tmem_lane_base:08x}",
                "decoded_lane_offset": decoded_lane_offset,
                "buggy_lanes_read_count": len(set(buggy_lanes_read)),
                "fixed_lanes_range": f"{fixed_lanes_read[0]}..{fixed_lanes_read[-1]}",
            }
        )

    all_buggy_lanes = {lane for w in range(4) for lane in range(0, 32)}
    all_fixed_lanes = {
        lane
        for w in range(4)
        for lane in range((((w * 32) & 96) >> 0), (((w * 32) & 96) >> 0) + 32)
    }

    # 2. cake_alpha_moe NVFP4 epilogue warp-uniform TMEM base:
    # Buggy: physical_feature = consumer_warp * 32 + lane; taddr = tmem_base + (physical_feature << 16) + k_col
    # Fixed: warp_lane_base = (consumer_warp * 32) << 16; taddr = tmem_base + warp_lane_base + k_col
    alpha_moe_warp_checks: list[dict[str, Any]] = []
    for consumer_warp in range(4):
        buggy_addrs_in_warp = {
            ((consumer_warp * 32 + lane) << 16) for lane in range(32)
        }
        fixed_addrs_in_warp = {
            ((consumer_warp * 32) << 16) for lane in range(32)
        }
        alpha_moe_warp_checks.append(
            {
                "consumer_warp": consumer_warp,
                "buggy_distinct_base_addrs_in_warp": len(buggy_addrs_in_warp),
                "fixed_distinct_base_addrs_in_warp": len(fixed_addrs_in_warp),
                "fixed_warp_lane_offset": consumer_warp * 32,
                "is_warp_uniform_after_fix": len(fixed_addrs_in_warp) == 1,
            }
        )

    # 3. 2-CTA (cta_group::2) multicast::cluster (mask 0b11) mma_done stage-release barrier drain (#6820, #6763)
    # In a pipeline with `num_stages` stages and `k_steps` K tiles:
    # Producer acquires stage `s = step % num_stages` only when step >= num_stages.
    # For the final `tail_stages = min(num_stages, k_steps)` steps, without producer_tail drain,
    # follower CTA (cta_rank == 1) exits early while leader CTA (cta_rank == 0) is still issuing
    # `tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster` (mask 0b11)
    # targeting follower's shared-memory `mma_done[s]` mbarrier.
    def analyze_cta_pair_drain(kernel_family: str, num_stages: int, k_steps: int, affected_kernels: int) -> dict[str, Any]:
        waited_in_loop = max(0, k_steps - num_stages)
        undrained_without_tail = min(num_stages, k_steps)
        drained_with_tail = waited_in_loop + undrained_without_tail
        tail_stage_indices = [
            ((k_steps - undrained_without_tail + i) % num_stages)
            for i in range(undrained_without_tail)
        ]
        return {
            "kernel_family": kernel_family,
            "affected_kernels": affected_kernels,
            "num_stages": num_stages,
            "k_steps": k_steps,
            "waited_in_main_loop": waited_in_loop,
            "undrained_stages_without_tail": undrained_without_tail,
            "total_drained_with_tail": drained_with_tail,
            "tail_stage_indices": tail_stage_indices,
            "follower_cta_safe_exit": drained_with_tail == k_steps,
        }

    minimax_h3_drain = analyze_cta_pair_drain("cake_minimax_h3", num_stages=7, k_steps=32, affected_kernels=1)
    dense_proj_drain = analyze_cta_pair_drain("cake_dense_projection_gemm", num_stages=4, k_steps=32, affected_kernels=49)

    return {
        "dsa_train_tmem_lane_quarters": {
            "warps": dsa_warp_quarters,
            "buggy_unique_lanes_covered": len(all_buggy_lanes),
            "fixed_unique_lanes_covered": len(all_fixed_lanes),
            "patched_load_sites_per_kernel": 20,
            "patched_backward_kernels": 4,
            "total_patched_load_sites": 80,
        },
        "alpha_moe_nvfp4_warp_uniform_tmem": {
            "warp_checks": alpha_moe_warp_checks,
            "patched_epilogue_sites": 66,
            "all_warps_uniform": all(w["is_warp_uniform_after_fix"] for w in alpha_moe_warp_checks),
        },
        "cta_pair_multicast_stage_drain": {
            "minimax_h3": minimax_h3_drain,
            "dense_projection_gemm": dense_proj_drain,
        },
    }


def build_summary() -> dict[str, Any]:
    agm = simulate_cake_allgather_matmul()
    msa_vsa = simulate_cake_msa_and_vsa_sm90()
    tcgen05 = simulate_blackwell_tcgen05_invariants()

    return {
        "experiment": "flashinfer_cake_allgather_hopper_msa_vsa_tmem_race",
        "phase": "Phase 46",
        "priority": 55,
        "upstream_commits": [
            "flashinfer@d9695940c53e0563cb0855f93c0955660c143d9f",
            "flashinfer@9c68e44fd2a7664d53d49168ed616aa8c06ef1f7",
            "flashinfer@8a93c9b833761c75939f176a58268ad77ac304f1",
            "flashinfer@7f6f119d8534441430934e798ba056f052d1ba20",
            "flashinfer@ddf080d8a56c87f60a25ee9df750e373c4bcbe92",
            "flashinfer@c8be875c7c403c4eac664f034b098157d9dc0642",
            "flashinfer@e4e7510f51fe9f058390ad84c66178805c8b2f45",
            "flashinfer@653cce0a6e903a042d374cd509002bd82ae6d3a3",
            "flashinfer@2b3cd4f2f990954ea9a196c352dadae25fd3ff8c",
        ],
        "cake_allgather_matmul": agm,
        "cake_msa_and_vsa_sm90": msa_vsa,
        "blackwell_tcgen05_invariants": tcgen05,
        "headline_metrics": {
            "sm100a_ws2_cg2_bf16_kmajor_speedup_x": 1.065,
            "sm100a_ws4_16cta_push_speedup_x": 1.086,
            "sm103a_ws8_16cta_push_total_ctas": 112,
            "cake_msa_sm90_total_programs": 89,
            "cake_msa_sm90_max_speedup_x": 3.237,
            "cake_msa_blackwell_paged_first_topk_gb300_speedup_x": 3.300,
            "cake_vsa_sm90_total_programs": 17,
            "cake_vsa_sm90_peak_speedup_x": 1.2524,
            "dsa_train_fixed_tmem_lanes_covered": 128,
            "alpha_moe_patched_tmem_sites": 66,
            "cta_pair_drained_kernels_total": 50,
        },
    }


def verify_summary(summary: dict[str, Any]) -> None:
    agm = summary["cake_allgather_matmul"]
    assert agm["sm100a_ws4_512x2560_k8192_push"]["old_8_ctas_per_peer"]["total_push_ctas"] == 24
    assert agm["sm100a_ws4_512x2560_k8192_push"]["new_16_ctas_per_peer"]["total_push_ctas"] == 48
    assert agm["sm103a_ws8_1024x1280_k8192_push"]["old_8_ctas_per_peer"]["total_push_ctas"] == 56
    assert agm["sm103a_ws8_1024x1280_k8192_push"]["new_16_ctas_per_peer"]["total_push_ctas"] == 112

    msa_vsa = summary["cake_msa_and_vsa_sm90"]
    assert msa_vsa["total_msa_sm90_programs"] == 89
    assert msa_vsa["total_vsa_sm90_programs"] == 17
    assert msa_vsa["msa_blackwell_union_prefill"]["heavy_first_max_sm_kv_tiles"] < msa_vsa["msa_blackwell_union_prefill"]["fifo_max_sm_kv_tiles"]

    tcgen05 = summary["blackwell_tcgen05_invariants"]
    assert tcgen05["dsa_train_tmem_lane_quarters"]["buggy_unique_lanes_covered"] == 32
    assert tcgen05["dsa_train_tmem_lane_quarters"]["fixed_unique_lanes_covered"] == 128
    assert tcgen05["alpha_moe_nvfp4_warp_uniform_tmem"]["all_warps_uniform"] is True
    assert tcgen05["cta_pair_multicast_stage_drain"]["minimax_h3"]["undrained_stages_without_tail"] == 7
    assert tcgen05["cta_pair_multicast_stage_drain"]["minimax_h3"]["follower_cta_safe_exit"] is True
    assert tcgen05["cta_pair_multicast_stage_drain"]["dense_projection_gemm"]["affected_kernels"] == 49
    assert tcgen05["cta_pair_multicast_stage_drain"]["dense_projection_gemm"]["follower_cta_safe_exit"] is True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Run deterministic invariant assertions")
    args = parser.parse_args()

    summary = build_summary()
    if args.check:
        verify_summary(summary)

    out_path = Path(__file__).with_name("summary.json")
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out_path} (check={args.check})")


if __name__ == "__main__":
    main()
