from __future__ import annotations

import json
from pathlib import Path


def evaluate_pd_layerwise_kv_transfer_tiers() -> dict:
    tiers = [
        {
            "tier": "Tier 1: End-of-Prefill Bulk Transfer + Uncoalesced Per-Layer-Per-Block RDMA (16,000 WQEs)",
            "post_prefill_kv_stall_ms": 148.0,
            "rnic_bandwidth_util_pct": 35.8,
            "rdma_wqes_per_request": 16000,
            "prefill_compute_overlap_pct": 0.0,
            "decision": "Send to a human",
        },
        {
            "tier": "Tier 2: End-of-Prefill Bulk Transfer + Coalesced Contiguous Slab Staging",
            "post_prefill_kv_stall_ms": 58.4,
            "rnic_bandwidth_util_pct": 91.2,
            "rdma_wqes_per_request": 80,
            "prefill_compute_overlap_pct": 0.0,
            "decision": "Send to a human",
        },
        {
            "tier": "Tier 3: Layerwise Pipelined Async Push Without Block Coalescing (200 WQEs/Layer)",
            "post_prefill_kv_stall_ms": 38.0,
            "rnic_bandwidth_util_pct": 42.5,
            "rdma_wqes_per_request": 16000,
            "prefill_compute_overlap_pct": 74.3,
            "decision": "Confirm",
        },
        {
            "tier": "Tier 4: Layerwise Pipelined Async Push + Per-Layer Coalesced Contiguous Pack (1 WQE/Layer)",
            "post_prefill_kv_stall_ms": 4.2,
            "rnic_bandwidth_util_pct": 94.6,
            "rdma_wqes_per_request": 80,
            "prefill_compute_overlap_pct": 92.8,
            "decision": "Act automatically",
        },
    ]

    context_scaling = [
        {"prompt_tokens": 4096, "bulk_uncoalesced_ms": 37.0, "bulk_coalesced_ms": 14.6, "layerwise_pipelined_ms": 1.2},
        {"prompt_tokens": 8192, "bulk_uncoalesced_ms": 74.0, "bulk_coalesced_ms": 29.2, "layerwise_pipelined_ms": 2.1},
        {"prompt_tokens": 16384, "bulk_uncoalesced_ms": 148.0, "bulk_coalesced_ms": 58.4, "layerwise_pipelined_ms": 4.2},
        {"prompt_tokens": 32768, "bulk_uncoalesced_ms": 296.0, "bulk_coalesced_ms": 116.8, "layerwise_pipelined_ms": 8.4},
    ]

    summary = {
        "benchmark": "llm_disaggregated_pd_layerwise_pipelined_kv_transfer_strided_block_packing_selection",
        "provenance": "synthetic + anonymized_reported (independently_reproduced=false, physical_gpu_cluster_measurement=not_measured)",
        "num_layers": 80,
        "prompt_tokens": 16384,
        "kv_blocks": 100,
        "naive_post_prefill_stall_ms": 148.0,
        "tier4_post_prefill_stall_ms": 4.2,
        "stall_reduction_x": round(148.0 / 4.2, 1),
        "tiers": tiers,
        "context_scaling": context_scaling,
    }
    return summary


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    results = evaluate_pd_layerwise_kv_transfer_tiers()
    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
