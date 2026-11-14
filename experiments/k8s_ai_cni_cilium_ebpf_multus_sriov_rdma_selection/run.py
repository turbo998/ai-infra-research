#!/usr/bin/env python3
"""Round 191: K8s AI Cluster CNI (HostNetwork vs Calico veth vs Multus IPVlan vs Cilium eBPF + SR-IOV RDMA) Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def evaluate_k8s_ai_cni_tiers(
    nic_line_rate_gbps: int = 400,
    k8s_services_count: int = 10000,
) -> dict[str, Any]:
    """Evaluate 4 Kubernetes AI container networking (CNI) & RDMA isolation tiers."""
    line_rate_gbs = nic_line_rate_gbps / 8.0  # 50.0 GB/s

    tiers = [
        {
            "tier_id": "tier1_hostnetwork_shared_root_netns",
            "name": "Tier 1: 裸开 hostNetwork: true (共享宿主机网络命名空间 · 零隔离)",
            "rdma_busbw_gbs": 48.4,
            "control_plane_rpc_p99_us": 420.0,
            "multi_tenant_port_conflict_rate_pct": 18.5,
            "hardware_rdma_netns_isolation": False,
            "verdict": "虽能跑满 RDMA 带宽，但多租户共享宿主机端口导致 MASTER_PORT=29500 频繁冲突，且 1 卡 Pod 可窥探全部 8 张网卡 ✕",
        },
        {
            "tier_id": "tier2_single_veth_iptables_overlay_soft_roce",
            "name": "Tier 2: 传统单平面 veth + iptables Overlay CNI (破坏硬件 RDMA 卸载)",
            "rdma_busbw_gbs": 2.1,
            "control_plane_rpc_p99_us": 680.0,
            "multi_tenant_port_conflict_rate_pct": 0.0,
            "hardware_rdma_netns_isolation": True,
            "verdict": "Linux veth 虚拟网桥无法透传硬件 RoCEv2 Verbs 队列，被迫退化为 TCP/Soft-RoCE，400G 带宽暴跌至 2.1 GB/s ✕",
        },
        {
            "tier_id": "tier3_multus_ipvlan_shared_rdma_dev",
            "name": "Tier 3: Multus 双平面 + IPVlan/Macvlan + Shared RDMA 共享设备模式",
            "rdma_busbw_gbs": 46.2,
            "control_plane_rpc_p99_us": 310.0,
            "multi_tenant_port_conflict_rate_pct": 0.0,
            "hardware_rdma_netns_isolation": False,
            "verdict": "实现了每 Pod 独立 IP 与 46.2 GB/s 带宽，但 `/dev/infiniband/uverbs*` 仍为共享模式，缺乏硬件队列级租户隔离 ✅",
        },
        {
            "tier_id": "tier4_cilium_ebpf_netkit_plus_multus_sriov_exclusive_rdma",
            "name": "Tier 4: Cilium eBPF (Netkit) 控制面 + Multus + SR-IOV 硬件 VF 独占 RDMA Netns",
            "rdma_busbw_gbs": 48.3,
            "control_plane_rpc_p99_us": 115.0,
            "multi_tenant_port_conflict_rate_pct": 0.0,
            "hardware_rdma_netns_isolation": True,
            "verdict": "eBPF 消除万级 Service iptables 开销 (115us)，SR-IOV VF + Exclusive RDMA Netns 跑满 48.3 GB/s 且 100% 硬件隔离 🏆",
        },
    ]

    return {
        "experiment_id": "k8s_ai_cni_cilium_ebpf_multus_sriov_rdma_selection",
        "round": 191,
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload": {
            "nic_line_rate_gbps": nic_line_rate_gbps,
            "line_rate_gbs": line_rate_gbs,
            "k8s_services_count": k8s_services_count,
            "veth_soft_roce_gbs": 2.1,
            "sriov_exclusive_gbs": 48.3,
        },
        "tiers": tiers,
    }


def main() -> None:
    results = evaluate_k8s_ai_cni_tiers()
    out_path = Path(__file__).resolve().parent / "results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
