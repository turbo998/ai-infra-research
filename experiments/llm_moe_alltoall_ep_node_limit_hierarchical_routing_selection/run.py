from __future__ import annotations
import json
from pathlib import Path


def evaluate_moe_node_limited_routing() -> dict:
    # Workload: 671B MoE (256 routed experts, Top-K=8, hidden=7168) across EP=64 (8 nodes x 8 GPUs, 400G RDMA = 50 GB/s, NVLink = 450 GB/s uni)
    tiers = [
        {
            "tier_id": "tier1_unconstrained_topk8_flat_alltoall_bf16",
            "name": "Unconstrained Top-8 Routing + Flat 64-Rank AllToAllv (BF16)",
            "rdma_copies_per_token": 7.12,
            "alltoall_latency_ms": 6.85,
            "overlap_compute_coverage_pct": 34.0,
            "action": "Send to a human",
            "status": "fail",
            "note": "Each token scatters to ~7.12 nodes over 50 GB/s RDMA in BF16 (6.85 ms/layer), far exceeding the 2.3 ms MLP compute window!"
        },
        {
            "tier_id": "tier2_m4_node_limited_routing_naive_flat_alltoall",
            "name": "M=4 Node-Limited Routing + Naive Flat 64-Rank ncclAllToAllv",
            "rdma_copies_per_token": 7.00,
            "alltoall_latency_ms": 6.60,
            "overlap_compute_coverage_pct": 35.5,
            "action": "Send to a human",
            "status": "fail",
            "note": "Restricts experts to <=4 nodes, but flat AllToAllv still sends separate RDMA packets to GPU 1 and GPU 5 on the same remote node (zero RDMA savings)!"
        },
        {
            "tier_id": "tier3_m4_node_limited_plus_2stage_rdma_nvlink_dedup_bf16",
            "name": "M=4 Node-Limited + 2-Stage Hierarchical RDMA->NVLink Deduplicated Dispatch (BF16)",
            "rdma_copies_per_token": 3.12,
            "alltoall_latency_ms": 2.65,
            "overlap_compute_coverage_pct": 86.0,
            "action": "Confirm",
            "status": "pass",
            "note": "Sends only 1 deduplicated copy per target node over RDMA (3.12 copies/tok, 2.28x less RDMA) and fans out over 900 GB/s intra-node NVLink."
        },
        {
            "tier_id": "tier4_m4_2stage_deepep_plus_fp8_dispatch_bf16_combine_sm_free",
            "name": "M=4 + DeepEP 2-Stage RDMA->NVLink Dedup + FP8 Dispatch / BF16 Combine (IBGDA)",
            "rdma_copies_per_token": 3.12,
            "alltoall_latency_ms": 1.42,
            "overlap_compute_coverage_pct": 100.0,
            "action": "Act automatically",
            "status": "recommended",
            "note": "Combines M=4 node deduplication (3.12 copies) with FP8 (1B) dispatch + BF16 combine and IBGDA overlap -> 1.42 ms (4.82x faster, 100% hidden under compute)!"
        },
    ]
    return {
        "experiment_id": "llm_moe_alltoall_ep_node_limit_hierarchical_routing_selection",
        "round": 287,
        "provenance": "synthetic + anonymized_reported (independently_reproduced=false, physical_gpu_cluster_measurement=not_measured)",
        "workload": {
            "routed_experts": 256,
            "top_k": 8,
            "ep_ranks": 64,
            "nodes": 8,
            "node_limit_m": 4
        },
        "tiers": tiers,
        "summary": {
            "rdma_volume_reduction_x": round((7.12 * 2) / (3.12 * 1), 2),
            "alltoall_speedup_x": round(6.85 / 1.42, 2),
            "recommended_latency_ms": 1.42,
            "recommended_tier": "tier4_m4_2stage_deepep_plus_fp8_dispatch_bf16_combine_sm_free"
        }
    }


if __name__ == "__main__":
    data = evaluate_moe_node_limited_routing()
    out = Path(__file__).parent / "results.json"
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out}")
