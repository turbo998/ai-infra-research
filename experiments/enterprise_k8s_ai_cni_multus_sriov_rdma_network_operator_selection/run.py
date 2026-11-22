from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_results() -> dict:
    cni_modes = [
        {
            "mode": "Tier 1: Standard K8s Overlay CNI Only (Calico/Flannel VXLAN over Linux Bridge veth, TCP Socket Fallback)",
            "achieved_nccl_busbw_gbps_400g": 4.2,
            "rdma_one_sided_latency_us": 42.0,
            "multi_tenant_network_namespace_isolation": True,
            "fractional_multi_pod_per_node_support": True,
        },
        {
            "mode": "Tier 2: hostNetwork=true + Privileged Container (Direct Host Root Netns Exposure)",
            "achieved_nccl_busbw_gbps_400g": 47.2,
            "rdma_one_sided_latency_us": 1.8,
            "multi_tenant_network_namespace_isolation": False,
            "fractional_multi_pod_per_node_support": False,
        },
        {
            "mode": "Tier 3: Multus CNI + Macvlan / IPvlan + RDMA Shared Device Plugin (Netns Isolated, Shared PF)",
            "achieved_nccl_busbw_gbps_400g": 45.8,
            "rdma_one_sided_latency_us": 2.1,
            "multi_tenant_network_namespace_isolation": True,
            "fractional_multi_pod_per_node_support": True,
        },
        {
            "mode": "Tier 4: NVIDIA Network Operator (Multus + SR-IOV VF + Topology Manager single-numa-node GPU-NIC PIX Alignment)",
            "achieved_nccl_busbw_gbps_400g": 47.6,
            "rdma_one_sided_latency_us": 1.8,
            "multi_tenant_network_namespace_isolation": True,
            "fractional_multi_pod_per_node_support": True,
        },
    ]
    return {
        "experiment_id": "enterprise_k8s_ai_cni_multus_sriov_rdma_network_operator_selection",
        "Provenance": "synthetic + anonymized_reported (independently_reproduced=false)",
        "nic_line_rate_gbps": 50.0,
        "cni_modes": cni_modes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    out_path = Path(__file__).resolve().parent / "results.json"
    payload = build_results()

    if args.check:
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        assert existing == payload, "results.json drift detected"
        print("OK: enterprise_k8s_ai_cni_multus_sriov_rdma_network_operator_selection verified.")
        return

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Wrote results.json")


if __name__ == "__main__":
    main()
