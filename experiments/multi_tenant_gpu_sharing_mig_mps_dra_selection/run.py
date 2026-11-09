from __future__ import annotations

import json
from pathlib import Path


def evaluate_gpu_sharing_tiers() -> dict:
    """Deterministic evaluator for 4 Multi-Tenant Fractional GPU Sharing & Isolation Tiers.

    Workload Baseline:
      - 56 Concurrent Enterprise AI Microservices / Tenants (8B LLM Agents, BGE-M3 Embedding, Cross-Encoder Rerankers, Jupyter Dev Pods; each requiring ~16-18 GB VRAM and ~12% peak compute)
      - Hardware Unit: NVIDIA H200 SXM (141 GB HBM3e, 132 SMs, up to 7 hardware MIG instances: 7x 1g.20gb)
      - Baseline Dedicated Allocation (No Sharing): Requires 56 full GPUs (average GPU utilization 18.2%)
      - Shared Packing Target: 7 Tenants per H200 GPU (8 GPUs total for 56 tenants; or 16 GPUs with headroom)
    """
    tiers = [
        {
            "tier_id": "tier1_k8s_time_slicing_oversubscription",
            "name": "Tier 1: Kubernetes Device Plugin Time-Slicing (OS Round-Robin Context Switching, No Memory Limit)",
            "name_zh": "Tier 1: K8s 原生时间片轮转 (Time-Slicing，零显存隔离 + 毫秒级上下文切换抖动)",
            "isolation_mechanism": "Temporal context-switching across 7 processes; shared HBM & L2 without quota enforcement",
            "aggregate_qps_per_gpu": 142.0,
            "noisy_neighbor_p99_latency_inflation_pct": 215.0,
            "oom_neighbor_kill_blast_radius_pct": 100.0,
            "illegal_mem_fault_blast_radius_pct": 0.0,
            "effective_gpu_packing_utilization_pct": 42.0,
            "gpus_required_for_56_services": 16,
            "annual_cluster_tco_usd": 384000,
        },
        {
            "tier_id": "tier2_cuda_mps_spatial_sm_sharing",
            "name": "Tier 2: NVIDIA CUDA MPS (Multi-Process Service Spatial SM Sharing + Shared GPU Page Table)",
            "name_zh": "Tier 2: NVIDIA CUDA MPS 空间算力并发 (共享单 GPU 页表上下文，存在致命故障连坐风险)",
            "isolation_mechanism": "Spatial concurrent SM execution + CUDA_MPS_PINNED_DEVICE_MEM_LIMIT, but shared GPU virtual address space",
            "aggregate_qps_per_gpu": 333.7,
            "noisy_neighbor_p99_latency_inflation_pct": 28.5,
            "oom_neighbor_kill_blast_radius_pct": 0.0,
            "illegal_mem_fault_blast_radius_pct": 100.0,
            "effective_gpu_packing_utilization_pct": 79.5,
            "gpus_required_for_56_services": 10,
            "annual_cluster_tco_usd": 240000,
        },
        {
            "tier_id": "tier3_hardware_mig_static_partitioning",
            "name": "Tier 3: Hardware MIG (Multi-Instance GPU: 7x 1g.20gb Physical Silicon Partitioning)",
            "name_zh": "Tier 3: 硬件级 MIG 物理切分 (7×1g.20gb 独立显存控制器/L2/SM 物理硬隔离)",
            "isolation_mechanism": "Dedicated SM slices, dedicated HBM memory channels, dedicated L2 banks & crossbar per instance",
            "aggregate_qps_per_gpu": 298.0,
            "noisy_neighbor_p99_latency_inflation_pct": 1.8,
            "oom_neighbor_kill_blast_radius_pct": 0.0,
            "illegal_mem_fault_blast_radius_pct": 0.0,
            "effective_gpu_packing_utilization_pct": 74.0,
            "gpus_required_for_56_services": 10,
            "annual_cluster_tco_usd": 240000,
        },
        {
            "tier_id": "tier4_k8s_dra_dynamic_mig_plus_hami_mps",
            "name": "Tier 4: K8s DRA + Dynamic Hardware MIG (Cross-Tenant Hard Boundary) + HAMi/MPS (Intra-Tenant Elastic Packing)",
            "name_zh": "Tier 4: K8s DRA 动态 MIG (跨租户物理隔离) + HAMi/MPS (同租户内弹性复用)",
            "isolation_mechanism": "Hardware MIG between security domains + CUDA API interception (HAMi libvgpu.so / MPS) inside trusted pods",
            "aggregate_qps_per_gpu": 352.0,
            "noisy_neighbor_p99_latency_inflation_pct": 3.2,
            "oom_neighbor_kill_blast_radius_pct": 0.0,
            "illegal_mem_fault_blast_radius_pct": 0.0,
            "effective_gpu_packing_utilization_pct": 88.5,
            "gpus_required_for_56_services": 8,
            "annual_cluster_tco_usd": 192000,
        },
    ]

    # Fault & Noisy Neighbor Perturbation Matrix across the 4 sharing modes
    fault_and_interference_matrix = [
        {
            "incident": "Noisy Neighbor Launches Heavy GEMM Flood (100% SM Attempt)",
            "time_slicing_impact": "+215.0% P99 Tail Latency Stall ✕",
            "cuda_mps_impact": "+28.5% L2 Cache & Memory Bandwidth Contention",
            "hardware_mig_impact": "+1.8% (Physically Isolated HBM Channels & L2) 🏆",
        },
        {
            "incident": "Runaway Tenant Allocates 130 GB VRAM (Memory Bomb)",
            "time_slicing_impact": "100% Neighbor Pods Crash with OOM ✕",
            "cuda_mps_impact": "0% Neighbor Impact (Blocked by MEM_LIMIT) ✅",
            "hardware_mig_impact": "0% Neighbor Impact (Hardware 20GB Partition) 🏆",
        },
        {
            "incident": "Buggy Custom Triton/CUDA Kernel Triggers Illegal Memory Access (XID 31)",
            "time_slicing_impact": "0% Neighbor Impact (Separate OS CUDA Contexts) ✅",
            "cuda_mps_impact": "100% Fatal Fate-Sharing! All 7 Pods Killed Together ✕",
            "hardware_mig_impact": "0% Neighbor Impact (Isolated Silicon Engines) 🏆",
        },
    ]

    t1 = tiers[0]
    t2 = tiers[1]
    t3 = tiers[2]
    t4 = tiers[3]
    dedicated_gpus = 28  # 2x per-GPU Naive or 28 GPUs baseline ($672,000/yr)

    summary = {
        "tier2_vs_tier1_throughput_gain_x": round(
            t2["aggregate_qps_per_gpu"] / t1["aggregate_qps_per_gpu"], 2
        ),
        "tier2_fault_blast_radius_pct": t2["illegal_mem_fault_blast_radius_pct"],
        "tier3_fault_blast_radius_pct": t3["illegal_mem_fault_blast_radius_pct"],
        "tier3_noisy_neighbor_p99_jitter_pct": t3["noisy_neighbor_p99_latency_inflation_pct"],
        "tier4_gpu_utilization_pct": t4["effective_gpu_packing_utilization_pct"],
        "tier4_vs_dedicated_gpu_saving_pct": round(
            (dedicated_gpus - t4["gpus_required_for_56_services"]) / dedicated_gpus * 100.0, 1
        ),
    }

    return {
        "experiment_id": "multi_tenant_gpu_sharing_mig_mps_dra_selection",
        "date": "2026-11-09",
        "provenance": {
            "type": "synthetic_multi_tenant_gpu_mig_mps_timeslicing_isolation_model_calibrated_to_anonymized_reported_specs",
            "independently_reproduced": False,
            "physical_gpu_cluster_measurement": "not_measured",
            "calibration_sources": [
                "NVIDIA Hopper H100/H200 Multi-Instance GPU (MIG 7x 1g.20gb) Silicon Crossbar & L2 Partitioning Architecture",
                "NVIDIA CUDA Multi-Process Service (MPS) Fatal Fault Propagation & Shared Page-Table Semantics",
                "CNCF HAMi (Heterogeneous AI Computing Virtualization Middleware) & Kubernetes 1.32+ DRA (Dynamic Resource Allocation)",
            ],
        },
        "workload_baseline": {
            "concurrent_microservices": 56,
            "gpu_sku": "NVIDIA H200 SXM (141 GB HBM3e, 7 MIG Slices)",
            "baseline_partially_packed_gpus": 28,
        },
        "tiers": tiers,
        "fault_and_interference_matrix": fault_and_interference_matrix,
        "summary": summary,
    }


def main() -> None:
    results = evaluate_gpu_sharing_tiers()
    out_path = Path(__file__).resolve().parent / "results.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
