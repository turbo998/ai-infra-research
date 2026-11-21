from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_payload() -> dict:
    return {
        "benchmark_id": "moe_architecture_deepep_ibgda_dualpipe_eplb_load_balancing_selection",
        "title": "MoE Systems Architecture (DeepEP Normal/IBGDA, DualPipe/Dual-Batch Overlap & EPLB Load Balancing) Benchmark",
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload_assumptions": {
            "model": "DeepSeek-V3/R1 671B MoE (256 Routed Experts, Top-8 Active)",
            "cluster_nodes_8x_h200": 8,
            "total_gpus_ep64": 64,
            "scaleout_fabric": "8x 400G RoCEv2 / InfiniBand RDMA",
        },
        "tiers": [
            {
                "tier_id": "tier1_standard_nccl_alltoall_without_overlap_or_eplb",
                "name": "Tier 1: Generic NCCL All-to-All (CPU Proxy, Serial Comm-Compute & Static Unbalanced Expert Placement)",
                "exposed_alltoall_communication_stall_pct": 54.8,
                "decode_alltoall_latency_us": 820.0,
                "hot_expert_straggler_skew_ratio": 3.85,
                "sla_compliant_goodput_tps_per_gpu": 520,
                "cost_per_1m_tokens_usd": 1.18,
                "verdict": "Anti-Pattern: 54.8% of wall-clock time spent waiting on blocking All-to-All & hot-expert stragglers",
            },
            {
                "tier_id": "tier2_deepep_kernels_without_dual_batch_overlap_or_eplb",
                "name": "Tier 2: DeepEP High-Throughput & IBGDA Low-Latency Kernels (Serial Execution Without Ping-Pong Overlap)",
                "exposed_alltoall_communication_stall_pct": 24.6,
                "decode_alltoall_latency_us": 195.0,
                "hot_expert_straggler_skew_ratio": 3.40,
                "sla_compliant_goodput_tps_per_gpu": 1180,
                "cost_per_1m_tokens_usd": 0.52,
                "verdict": "Cuts raw All-to-All latency 4.2x via IBGDA, but serial dispatch/combine still leaves SMs idle 24.6% of step",
            },
            {
                "tier_id": "tier3_deepep_plus_dual_batch_overlap_without_dynamic_eplb",
                "name": "Tier 3: DeepEP + Two-Batch Ping-Pong Overlap (Hides Comm Behind Compute, Static Expert Placement)",
                "exposed_alltoall_communication_stall_pct": 6.8,
                "decode_alltoall_latency_us": 182.0,
                "hot_expert_straggler_skew_ratio": 2.95,
                "sla_compliant_goodput_tps_per_gpu": 1760,
                "cost_per_1m_tokens_usd": 0.35,
                "verdict": "Hides 88% of RDMA transfer time, but domain-shift traffic spikes overload single-GPU hot experts (2.95x skew)",
            },
            {
                "tier_id": "tier4_deepep_ibgda_dual_batch_overlap_plus_hierarchical_eplb",
                "name": "Tier 4: DeepEP (Normal + IBGDA) + DualPipe/Two-Batch Overlap + Hierarchical EPLB Redundant Hot Experts",
                "exposed_alltoall_communication_stall_pct": 1.8,
                "decode_alltoall_latency_us": 168.0,
                "hot_expert_straggler_skew_ratio": 1.09,
                "sla_compliant_goodput_tps_per_gpu": 2380,
                "cost_per_1m_tokens_usd": 0.25,
                "verdict": "Recommended: 1.8% exposed comm stall, 168us IBGDA latency, 1.09x load balance & $0.25/1M (-78.8%)",
            },
        ],
        "headline_findings": {
            "exposed_comm_stall_pct": 1.8,
            "decode_alltoall_latency_us": 168.0,
            "goodput_tps_per_gpu": 2380,
            "cost_reduction_vs_tier1_pct": 78.8,
        },
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
            raise SystemExit("results.json does not exist")
        if out_path.read_text(encoding="utf-8") != rendered:
            raise SystemExit("results.json is out of date; re-run run.py")
        return 0

    out_path.write_text(rendered, encoding="utf-8")
    print(f"Wrote {out_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
