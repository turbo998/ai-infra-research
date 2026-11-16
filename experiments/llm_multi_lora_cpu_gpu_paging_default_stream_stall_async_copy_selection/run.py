from __future__ import annotations

import json
from pathlib import Path


def evaluate_lora_paging_tiers() -> dict:
    tiers = [
        {
            "tier": "Tier 1: Pageable RAM / Disk Load on Default Compute Stream (Synchronous Admit)",
            "decode_stall_on_miss_ms": 64.00,
            "p99_tpot_under_churn_ms": 78.20,
            "pcie_bandwidth_gbps": 5.0,
            "cudagraph_compatible": False,
            "decision": "Send to a human",
        },
        {
            "tier": "Tier 2: Pinned Host Cache (max_cpu_loras) Copied on Compute Stream Before Step",
            "decode_stall_on_miss_ms": 11.50,
            "p99_tpot_under_churn_ms": 25.70,
            "pcie_bandwidth_gbps": 27.8,
            "cudagraph_compatible": True,
            "decision": "Send to a human",
        },
        {
            "tier": "Tier 3: Layerwise Async H2D Copy Stream Without LoRA-Affinity Batch Grouping",
            "decode_stall_on_miss_ms": 2.40,
            "p99_tpot_under_churn_ms": 16.60,
            "pcie_bandwidth_gbps": 51.2,
            "cudagraph_compatible": True,
            "decision": "Confirm",
        },
        {
            "tier": "Tier 4: Pinned LRU + Background H2D Stream Stage-Before-Admit + Affinity Grouping",
            "decode_stall_on_miss_ms": 0.08,
            "p99_tpot_under_churn_ms": 14.28,
            "pcie_bandwidth_gbps": 54.0,
            "cudagraph_compatible": True,
            "decision": "Act automatically",
        },
    ]

    rank_scaling = [
        {"lora_rank": 16, "adapter_mb": 80, "pageable_stall_ms": 16.0, "stage_before_admit_ms": 0.05},
        {"lora_rank": 32, "adapter_mb": 160, "pageable_stall_ms": 32.0, "stage_before_admit_ms": 0.06},
        {"lora_rank": 64, "adapter_mb": 320, "pageable_stall_ms": 64.0, "stage_before_admit_ms": 0.08},
        {"lora_rank": 128, "adapter_mb": 640, "pageable_stall_ms": 128.0, "stage_before_admit_ms": 0.10},
    ]

    summary = {
        "benchmark": "llm_multi_lora_cpu_gpu_paging_default_stream_stall_async_copy_selection",
        "provenance": "synthetic + anonymized_reported (independently_reproduced=false, physical_gpu_cluster_measurement=not_measured)",
        "lora_rank": 64,
        "adapter_size_mb": 320,
        "gpu_slots_max_loras": 8,
        "cpu_slots_max_cpu_loras": 64,
        "naive_decode_stall_ms": 64.00,
        "tier4_decode_stall_ms": 0.08,
        "stall_reduction_x": round(64.00 / 0.08, 1),
        "tiers": tiers,
        "rank_scaling": rank_scaling,
    }
    return summary


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    results = evaluate_lora_paging_tiers()
    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
