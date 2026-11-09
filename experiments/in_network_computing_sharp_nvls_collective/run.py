from __future__ import annotations

import json
from pathlib import Path


def evaluate_in_network_collective_tiers() -> dict:
    """Deterministic evaluator for 4 AI Cluster Collective Communication & In-Network Computing Tiers.

    Workload Baseline:
      - 2,048-GPU H200 SXM Cluster (256 nodes × 8 GPUs) with 8×400Gbps NDR InfiniBand / RoCEv2 NICs per node
        (Theoretical Aggregate Per-Node Unidirectional Line Rate = 400 GB/s)
      - Collective Payload: 256 MB BF16 Gradient / Activation AllReduce across 2,048 GPUs
      - Evaluates:
        1) NCCL Algorithm Bandwidth (`algbw` in GB/s, where higher means faster completion of 256MB AllReduce)
        2) 256MB AllReduce Completion Latency (ms) at 2,048 GPUs
        3) Network Bytes Transmitted per GPU Relative to Payload Size (`2.0x` for Ring/Tree vs `1.0x` for In-Switch SHARP)
        4) GPU SMs Occupied by NCCL Communication Kernels (Out of 132 H200 SMs — fewer SMs means higher GEMM overlap!)
        5) End-to-End 2,048-GPU LLM Training Scaling Efficiency / MFU (%)
    """
    tiers = [
        {
            "tier_id": "tier1_nccl_ring_allreduce",
            "name": "Tier 1: Standard NCCL Ring AllReduce (2(N-1) Sequential Hops, 2.0x Network Traffic)",
            "name_zh": "Tier 1: 传统 NCCL Ring 环状 AllReduce (2(N-1) 跳线性延迟墙 + 2倍流量放大)",
            "collective_mechanism": "Logical ring across 2,048 GPUs: Reduce-Scatter (1.0x bytes) + All-Gather (1.0x bytes)",
            "algbw_gbps_at_2048_gpus": 200.0,
            "allreduce_256mb_latency_ms": 14.8,
            "network_traffic_multiplier_x": 2.0,
            "gpu_sms_consumed_by_nccl": 20,
            "cluster_2048_gpu_mfu_pct": 43.5,
            "cross_campus_10ms_wan_mfu_pct": 11.2,
        },
        {
            "tier_id": "tier2_nccl_double_binary_tree",
            "name": "Tier 2: NCCL Double Binary Tree AllReduce (O(log N) Hops, Host-GPU Reduction)",
            "name_zh": "Tier 2: NCCL 双二叉树 (Double Binary Tree) AllReduce (对数跳数 + 仍占 20 个 SM)",
            "collective_mechanism": "Two complementary binary trees over nodes; O(log2 N) latency but still 2.0x NIC traffic & GPU SM add",
            "algbw_gbps_at_2048_gpus": 258.0,
            "allreduce_256mb_latency_ms": 6.4,
            "network_traffic_multiplier_x": 2.0,
            "gpu_sms_consumed_by_nccl": 20,
            "cluster_2048_gpu_mfu_pct": 49.2,
            "cross_campus_10ms_wan_mfu_pct": 16.5,
        },
        {
            "tier_id": "tier3_nvls_plus_ib_sharpv3_in_switch",
            "name": "Tier 3: Intra-Node NVSwitch NVLS + Inter-Node Quantum-2/3 IB SHARPv3 In-Switch Aggregation",
            "name_zh": "Tier 3: 机内 NVSwitch NVLS + 跨机 IB SHARPv3 交换机网内硬件规约 (流量砍半 + 释放 16 个 SM)",
            "collective_mechanism": "NVSwitch ASIC ALU reduces 8 local GPUs + Leaf/Spine IB Switch ASIC reduces across racks (1.0x traffic!)",
            "algbw_gbps_at_2048_gpus": 368.0,
            "allreduce_256mb_latency_ms": 3.2,
            "network_traffic_multiplier_x": 1.0,
            "gpu_sms_consumed_by_nccl": 4,
            "cluster_2048_gpu_mfu_pct": 56.8,
            "cross_campus_10ms_wan_mfu_pct": 21.0,
        },
        {
            "tier_id": "tier4_hierarchical_sharp_plus_diloco_cross_dc",
            "name": "Tier 4: Hierarchical Communicators (TP=NVLS, DP=IB SHARP) + Cross-DC Low-Freq Outer-Momentum (DiLoCo H=50)",
            "name_zh": "Tier 4: 分层通信域隔离 (TP用NVLS / DP用SHARP) + 跨机房低频外动量同步 (Hierarchical DiLoCo)",
            "collective_mechanism": "Isolates SHARP trees strictly to DP groups (prevents switch tree exhaustion) + H=50 outer sync across DCs",
            "algbw_gbps_at_2048_gpus": 375.0,
            "allreduce_256mb_latency_ms": 3.1,
            "network_traffic_multiplier_x": 1.0,
            "gpu_sms_consumed_by_nccl": 4,
            "cluster_2048_gpu_mfu_pct": 57.6,
            "cross_campus_10ms_wan_mfu_pct": 53.8,
        },
    ]

    # Cluster Scale Sensitivity (64 -> 256 -> 1,024 -> 4,096 GPUs) for 256MB AllReduce Latency (ms)
    scale_latency_perturbation = [
        {"gpus": 64, "ring_ms": 1.9, "tree_ms": 1.7, "nvls_sharp_ms": 1.1},
        {"gpus": 256, "ring_ms": 3.8, "tree_ms": 2.9, "nvls_sharp_ms": 1.7},
        {"gpus": 1024, "ring_ms": 8.2, "tree_ms": 4.6, "nvls_sharp_ms": 2.4},
        {"gpus": 2048, "ring_ms": 14.8, "tree_ms": 6.4, "nvls_sharp_ms": 3.2},
        {"gpus": 4096, "ring_ms": 28.5, "tree_ms": 8.9, "nvls_sharp_ms": 3.9},
    ]

    t1 = tiers[0]
    t3 = tiers[2]
    t4 = tiers[3]

    summary = {
        "tier3_vs_tier1_algbw_gain_x": round(
            t3["algbw_gbps_at_2048_gpus"] / t1["algbw_gbps_at_2048_gpus"], 2
        ),
        "tier3_vs_tier1_latency_speedup_x": round(
            t1["allreduce_256mb_latency_ms"] / t3["allreduce_256mb_latency_ms"], 2
        ),
        "tier3_network_traffic_reduction_pct": round(
            (t1["network_traffic_multiplier_x"] - t3["network_traffic_multiplier_x"])
            / t1["network_traffic_multiplier_x"]
            * 100.0,
            1,
        ),
        "tier3_sm_saved_per_gpu": t1["gpu_sms_consumed_by_nccl"] - t3["gpu_sms_consumed_by_nccl"],
        "tier4_cross_dc_mfu_gain_pt": round(
            t4["cross_campus_10ms_wan_mfu_pct"] - t1["cross_campus_10ms_wan_mfu_pct"], 1
        ),
    }

    return {
        "experiment_id": "in_network_computing_sharp_nvls_collective",
        "date": "2026-11-09",
        "provenance": {
            "type": "synthetic_nccl_ring_tree_nvls_sharp_collective_and_diloco_model_calibrated_to_anonymized_reported_specs",
            "independently_reproduced": False,
            "physical_gpu_cluster_measurement": "not_measured",
            "calibration_sources": [
                "NVIDIA NCCL 2.21+ NVLS (NVLink SHARP) & Quantum-2/3 InfiniBand SHARPv3 In-Switch Reduction Architecture",
                "Ring AllReduce 2(N-1)/N Bandwidth Amplification vs In-Switch 1.0x Tree Aggregation Law",
                "DeepMind DiLoCo / Hierarchical Low-Communication Data Parallelism over Multi-Campus Links",
            ],
        },
        "workload_baseline": {
            "cluster_gpus": 2048,
            "node_interconnect": "NVSwitch 3.0 (900 GB/s) + 8x400Gbps NDR InfiniBand (400 GB/s per node)",
            "collective_payload_mb": 256,
        },
        "tiers": tiers,
        "scale_latency_perturbation": scale_latency_perturbation,
        "summary": summary,
    }


def main() -> None:
    results = evaluate_in_network_collective_tiers()
    out_path = Path(__file__).resolve().parent / "results.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
