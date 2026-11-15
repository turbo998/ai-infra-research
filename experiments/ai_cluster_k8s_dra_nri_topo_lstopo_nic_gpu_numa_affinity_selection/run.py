#!/usr/bin/env python3
"""Round 269: Kubernetes Multi-GPU & Multi-RNIC Topology-Aware Allocation (Device Plugin vs Topology Manager single-numa-node vs DRA PIX Co-Allocation) Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def evaluate_k8s_gpu_nic_topology_dra_tiers() -> dict[str, Any]:
    """Evaluate 4 Kubernetes multi-GPU + multi-RNIC device allocation & topology affinity architectures."""
    tiers = [
        {
            "tier_id": "tier1_legacy_independent_device_plugins_topology_none",
            "name": "Tier 1: 默认独立 Device Plugin (topology=none，GPU 插件选 NUMA0 的 gpu0 而 RDMA 插件选 NUMA1 的 mlx5_7，触发 SYS 跨 CPU 绕行)",
            "effective_gdr_bw_per_nic_gbps": 11.2,
            "full_node_8gpu_pod_admission_rate_pct": 100.0,
            "pix_1to1_gpu_nic_match_rate_pct": 12.5,
            "verdict": "GPU 插件与 RDMA 插件互不通气，把 GPU0 与远端 CPU 插槽下的网卡配成一对，跨 UPI/XGMI 总线带宽暴跌 76.4% ✕",
        },
        {
            "tier_id": "tier2_kubelet_topology_manager_single_numa_node_policy",
            "name": "Tier 2: Kubelet Topology Manager (single-numa-node，虽锁死同 NUMA 但仍混配不同 PCIe Switch PHB，且 100% 误杀 8 卡 Pod!)",
            "effective_gdr_bw_per_nic_gbps": 21.5,
            "full_node_8gpu_pod_admission_rate_pct": 0.0,
            "pix_1to1_gpu_nic_match_rate_pct": 50.0,
            "verdict": "同一个 NUMA 下有 2 颗 PCIe Switch，仍会错配成 PHB 绕行 CPU Root Complex；更致命的是 8 卡 Pod 因跨双 NUMA 100% 报 TopologyAffinityError 拒载 ✕",
        },
        {
            "tier_id": "tier3_static_per_switch_resource_naming_manual_yaml_binding",
            "name": "Tier 3: 静态按 PCIe Switch 拆分资源名 (nvidia.com/gpu_sw0 + rdma/sw0，虽能对齐 PIX 但破坏统一调度池产生碎片)",
            "effective_gdr_bw_per_nic_gbps": 47.5,
            "full_node_8gpu_pod_admission_rate_pct": 100.0,
            "pix_1to1_gpu_nic_match_rate_pct": 100.0,
            "verdict": "逼迫业务在 YAML 里硬编码物理交换芯片编号 (sw0..sw3)，一旦 sw0 满而 sw1 空闲，Pod 永远 Pending 无法自动调度 ✅",
        },
        {
            "tier_id": "tier4_k8s_dra_dynamic_resource_allocation_or_cdi_pix_co_allocation",
            "name": "Tier 4: Kubernetes DRA (Dynamic Resource Allocation) / NRI 拓扑感知联合分配 (按 pcie_switch_id 1:1 绑定 PIX GPU+RNIC)",
            "effective_gdr_bw_per_nic_gbps": 47.5,
            "full_node_8gpu_pod_admission_rate_pct": 100.0,
            "pix_1to1_gpu_nic_match_rate_pct": 100.0,
            "verdict": "在单个 ResourceClaim 内用 CEL 约束 GPU 与 RNIC 共享同一颗 PCIe Switch (100% PIX 47.5GB/s)，且完美兼容 1/2/4/8 卡任意规格 Pod 🏆",
        },
    ]

    return {
        "experiment_id": "ai_cluster_k8s_dra_nri_topo_lstopo_nic_gpu_numa_affinity_selection",
        "round": 269,
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload": {
            "node_topology": "2 NUMA Sockets, 4 PCIe Gen5 Switches, 8 GPUs, 8x400G RNICs",
            "bw_speedup_vs_blind_plugin_x": round(47.5 / 11.2, 2),
        },
        "tiers": tiers,
    }


def main() -> None:
    results = evaluate_k8s_gpu_nic_topology_dra_tiers()
    out_path = Path(__file__).resolve().parent / "results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
