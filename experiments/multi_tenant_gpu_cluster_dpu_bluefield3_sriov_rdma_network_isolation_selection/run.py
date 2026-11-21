from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_payload() -> dict:
    return {
        "experiment_id": "multi_tenant_gpu_cluster_dpu_bluefield3_sriov_rdma_network_isolation_selection",
        "title": "Multi-Tenant GPU Cluster Network Isolation & DPU Offload: BlueField-3 DPU vs. SR-IOV + Multus CNI vs. IB P_Key / RoCEv2 Per-VF QoS",
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload_basis": {
            "cluster_size_gpus": 1024,
            "fabric_speed_gbps": 400,
            "co_located_tenants": 12,
            "noisy_neighbor_event": "Tenant A AlltoAll Incast & Slow-Receiver PFC Pause Storm",
        },
        "tiers": [
            {
                "tier": "Tier 1: Un-Isolated hostNetwork: true on Shared Lossless Queue (or Software VXLAN)",
                "gpudirect_rdma_busbw_gbps": 340.0,
                "noisy_neighbor_victim_busbw_drop_pct": 62.0,
                "host_cpu_cores_consumed_by_sdn": 18,
                "zero_trust_control_plane_isolation": False,
                "recommended_scenario": "Anti-pattern for multi-tenant clusters (PFC pause storm stalls all co-located tenants)",
            },
            {
                "tier": "Tier 2: Multus CNI + SR-IOV VF + Per-Tenant VLAN Isolation (Shared ToR Buffer Pool)",
                "gpudirect_rdma_busbw_gbps": 365.0,
                "noisy_neighbor_victim_busbw_drop_pct": 28.5,
                "host_cpu_cores_consumed_by_sdn": 8,
                "zero_trust_control_plane_isolation": False,
                "recommended_scenario": "Internal trusted multi-team K8s cluster with cooperative workloads",
            },
            {
                "tier": "Tier 3: SR-IOV + IB P_Key / RoCEv2 Per-VF Hardware Rate Pacing + ToR Dynamic Buffer",
                "gpudirect_rdma_busbw_gbps": 376.0,
                "noisy_neighbor_victim_busbw_drop_pct": 4.2,
                "host_cpu_cores_consumed_by_sdn": 4,
                "zero_trust_control_plane_isolation": False,
                "recommended_scenario": "Multi-department enterprise AI cluster requiring strict RDMA SLA protection",
            },
            {
                "tier": "Tier 4 (Recommended): NVIDIA BlueField-3 DPU / SuperNIC (DOCA VPC + Hardware RTT CC + PFC Watchdog)",
                "gpudirect_rdma_busbw_gbps": 388.0,
                "noisy_neighbor_victim_busbw_drop_pct": 0.8,
                "host_cpu_cores_consumed_by_sdn": 0,
                "zero_trust_control_plane_isolation": True,
                "recommended_scenario": "Multi-tenant AI Cloud / Sovereign AI Factory requiring bare-metal zero-trust & <1% jitter",
            },
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    out_path = Path(__file__).resolve().parent / "results.json"
    payload = build_payload()
    rendered = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"

    if args.check:
        if not out_path.exists():
            raise SystemExit("results.json missing")
        if out_path.read_text(encoding="utf-8") != rendered:
            raise SystemExit("results.json out of date")

    out_path.write_text(rendered, encoding="utf-8")
    print("Wrote results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
