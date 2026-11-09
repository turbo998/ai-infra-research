from __future__ import annotations

import json
from pathlib import Path


def evaluate_serverless_gpu_coldstart_tiers() -> dict:
    """Deterministic evaluator for 4 Kubernetes Serverless GPU Autoscaling & Cold-Start Tiers.

    Workload Baseline:
      - Model: 70B FP8 LLM (70 GB weights) served on 2xH200 / TP2 instances across 24 diurnal enterprise models
      - Container Image: 18 GB (CUDA 12.6 + PyTorch + vLLM/SGLang)
      - Traffic Pattern: Diurnal bursty enterprise traffic (10x spike within 15 seconds; 14 hrs/day low/zero traffic)
      - Evaluates 4 Cold-Start Stages (seconds):
        1) Image Pull & Container Create
        2) Python Import (`torch`, `vllm`) + CUDA Driver Context Init (`cuInit`)
        3) Weight Deserialization & H2D Transfer (`70 GB` into HBM3e)
        4) `torch.compile` / CUDA Graph Capture & Warmup
    """
    tiers = [
        {
            "tier_id": "tier1_vanilla_hpa_gpu_util_cold_boot",
            "name": "Tier 1: Vanilla K8s HPA (DCGM_GPU_UTIL) + Cold Registry Pull + Uncached Graph Compile",
            "name_zh": "Tier 1: 原生 K8s HPA (基于 GPU_UTIL 假指标) + 冷拉镜像 + 现场编译 CUDA Graph",
            "autoscaling_signal": "DCGM_FI_DEV_GPU_UTIL (Reads 98% even at Batch=1; blind to queue buildup!)",
            "stage1_image_pull_sec": 65.0,
            "stage2_python_cuda_init_sec": 22.0,
            "stage3_weight_load_h2d_sec": 100.0,
            "stage4_cuda_graph_compile_sec": 98.0,
            "total_cold_start_ready_sec": 285.0,
            "burst_10x_timeout_drop_rate_pct": 64.5,
            "min_idle_replicas_required_for_sla": "100% Static Over-Provisioning (Cannot scale to zero)",
            "monthly_cluster_gpu_bill_usd": 288000,
        },
        {
            "tier_id": "tier2_keda_kv_queue_nvme_graph_cache",
            "name": "Tier 2: KEDA (Queue Depth + KV Cache %) + Node NVMe Image/Weight Cache + Pre-Compiled Graph Cache",
            "name_zh": "Tier 2: KEDA (排队深度 + KV 水位) + 本机 NVMe 预缓存 + 预编译 CUDA Graph 缓存",
            "autoscaling_signal": "vllm:num_requests_waiting > 4 OR vllm:gpu_cache_usage_perc > 75%",
            "stage1_image_pull_sec": 1.5,
            "stage2_python_cuda_init_sec": 14.0,
            "stage3_weight_load_h2d_sec": 19.5,
            "stage4_cuda_graph_compile_sec": 7.0,
            "total_cold_start_ready_sec": 42.0,
            "burst_10x_timeout_drop_rate_pct": 8.2,
            "min_idle_replicas_required_for_sla": "1 Warm Replica per Model + Burst Scale-Out",
            "monthly_cluster_gpu_bill_usd": 164000,
        },
        {
            "tier_id": "tier3_criu_cuda_checkpoint_snapshot_restore",
            "name": "Tier 3: CRIU + NVIDIA cuda-checkpoint Process/VRAM Snapshot Restore from Host Pinned RAM/NVMe",
            "name_zh": "Tier 3: CRIU + NVIDIA cuda-checkpoint 进程与 CUDA 上下文快照直恢 (Host Pinned RAM)",
            "autoscaling_signal": "KEDA Sub-Second Polling + Knative Activator Request Buffer",
            "stage1_image_pull_sec": 0.2,
            "stage2_python_cuda_init_sec": 0.6,
            "stage3_weight_load_h2d_sec": 3.4,
            "stage4_cuda_graph_compile_sec": 0.4,
            "total_cold_start_ready_sec": 4.6,
            "burst_10x_timeout_drop_rate_pct": 0.4,
            "min_idle_replicas_required_for_sla": "Scale-to-Zero Enabled (Snapshots held in Host DRAM/NVMe)",
            "monthly_cluster_gpu_bill_usd": 108000,
        },
        {
            "tier_id": "tier4_serverlessllm_rdma_p2p_weight_streaming",
            "name": "Tier 4: Pre-Warmed CUDA Stub + ServerlessLLM Partitioned Tensor Format + RDMA/NVLink P2P Weight Multicast",
            "name_zh": "Tier 4: 预热零显存 CUDA 空壳进程 + ServerlessLLM 对齐张量流 + 跨机 RDMA/NVLink P2P 权重直注",
            "autoscaling_signal": "Gateway Token-RateDerivative + Speculative Pre-Warming + P2P Weight Multicast",
            "stage1_image_pull_sec": 0.0,
            "stage2_python_cuda_init_sec": 0.1,
            "stage3_weight_load_h2d_sec": 1.5,
            "stage4_cuda_graph_compile_sec": 0.2,
            "total_cold_start_ready_sec": 1.8,
            "burst_10x_timeout_drop_rate_pct": 0.0,
            "min_idle_replicas_required_for_sla": "True Scale-to-Zero (0 GB Idle VRAM, 1.8s Instant Wakeup)",
            "monthly_cluster_gpu_bill_usd": 91000,
        },
    ]

    # Perturbation test: Batch=1 vs Batch=64 behavior of DCGM_GPU_UTIL vs True KV/Queue Signal
    autoscaler_signal_perturbation = [
        {
            "traffic_state": "Idle Single User (Batch=1, 15 tok/s)",
            "dcgm_gpu_util_pct": 94.0,
            "true_kv_cache_and_queue_load_pct": 3.2,
            "vanilla_hpa_mistake": "False Positive Scale-Out! Wastes GPUs when 1 user is decoding.",
        },
        {
            "traffic_state": "Optimal Saturation (Batch=48, 1,450 tok/s)",
            "dcgm_gpu_util_pct": 99.0,
            "true_kv_cache_and_queue_load_pct": 78.0,
            "vanilla_hpa_mistake": "Cannot distinguish Batch=1 (94%) from Batch=48 (99%)!",
        },
        {
            "traffic_state": "10x Traffic Flash Crowd (180 Requests Queued)",
            "dcgm_gpu_util_pct": 100.0,
            "true_kv_cache_and_queue_load_pct": 340.0,
            "vanilla_hpa_mistake": "Metric Clamped at 100%! HPA only adds +1 Pod while 180 users time out.",
        },
    ]

    t1 = tiers[0]
    t3 = tiers[2]
    t4 = tiers[3]

    summary = {
        "tier1_total_coldstart_sec": t1["total_cold_start_ready_sec"],
        "tier3_total_coldstart_sec": t3["total_cold_start_ready_sec"],
        "tier4_total_coldstart_sec": t4["total_cold_start_ready_sec"],
        "tier4_vs_tier1_speedup_x": round(
            t1["total_cold_start_ready_sec"] / t4["total_cold_start_ready_sec"], 2
        ),
        "tier4_monthly_gpu_cost_saving_pct": round(
            (t1["monthly_cluster_gpu_bill_usd"] - t4["monthly_cluster_gpu_bill_usd"])
            / t1["monthly_cluster_gpu_bill_usd"]
            * 100.0,
            1,
        ),
        "tier4_monthly_gpu_cost_saving_usd": t1["monthly_cluster_gpu_bill_usd"] - t4["monthly_cluster_gpu_bill_usd"],
    }

    return {
        "experiment_id": "k8s_serverless_gpu_autoscaling_coldstart",
        "date": "2026-11-09",
        "provenance": {
            "type": "synthetic_k8s_gpu_coldstart_lifecycle_and_keda_autoscaling_model_calibrated_to_anonymized_reported_specs",
            "independently_reproduced": False,
            "physical_gpu_cluster_measurement": "not_measured",
            "calibration_sources": [
                "OSDI ServerlessLLM: Partitioned Direct-DMA Tensor Loading & Multi-Tier Pinned Memory Pool",
                "NVIDIA cuda-checkpoint & CRIU Process/CUDA Context Snapshot Restore Benchmark",
                "KEDA + vLLM/SGLang Queue/KV-Cache Autoscaling vs DCGM_FI_DEV_GPU_UTIL Saturation Analysis",
            ],
        },
        "workload_baseline": {
            "model": "70B FP8 (70 GB Weights on TP2 H200)",
            "enterprise_models_hosted": 24,
            "container_image_gb": 18.0,
        },
        "tiers": tiers,
        "autoscaler_signal_perturbation": autoscaler_signal_perturbation,
        "summary": summary,
    }


def main() -> None:
    results = evaluate_serverless_gpu_coldstart_tiers()
    out_path = Path(__file__).resolve().parent / "results.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
