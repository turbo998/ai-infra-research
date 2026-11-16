from __future__ import annotations

import json
from pathlib import Path


def evaluate_nccl_commsplit_tiers() -> dict:
    tiers = [
        {
            "tier": "Tier 1: Sequential dist.new_group() Loop + Separate Pairwise P2P Comms",
            "cluster_init_latency_sec": 1152.0,
            "nccl_hbm_per_gpu_gb": 14.80,
            "active_comms_per_rank": 18,
            "tcpstore_unique_id_exchanges": 3456,
            "usable_training_hbm_gb": 65.20,
            "decision": "Send to a human",
        },
        {
            "tier": "Tier 2: Lazy Communicator Init + Unified PP batch_isend_irecv Comm",
            "cluster_init_latency_sec": 340.0,
            "nccl_hbm_per_gpu_gb": 7.60,
            "active_comms_per_rank": 8,
            "tcpstore_unique_id_exchanges": 1408,
            "usable_training_hbm_gb": 72.40,
            "decision": "Send to a human",
        },
        {
            "tier": "Tier 3: Parallel dist.split_group (ncclCommSplit, splitShare=0)",
            "cluster_init_latency_sec": 24.0,
            "nccl_hbm_per_gpu_gb": 6.40,
            "active_comms_per_rank": 7,
            "tcpstore_unique_id_exchanges": 7,
            "usable_training_hbm_gb": 73.60,
            "decision": "Confirm",
        },
        {
            "tier": "Tier 4: Parallel ncclCommSplit (splitShare=1) + Dimension-Tuned Channels",
            "cluster_init_latency_sec": 18.5,
            "nccl_hbm_per_gpu_gb": 1.85,
            "active_comms_per_rank": 7,
            "tcpstore_unique_id_exchanges": 7,
            "usable_training_hbm_gb": 78.15,
            "decision": "Act automatically",
        },
    ]

    scale_points = [
        {"gpus": 512, "new_group_loop_sec": 95.0, "commsplit_shared_sec": 4.2},
        {"gpus": 1024, "new_group_loop_sec": 220.0, "commsplit_shared_sec": 7.5},
        {"gpus": 2048, "new_group_loop_sec": 510.0, "commsplit_shared_sec": 12.1},
        {"gpus": 4096, "new_group_loop_sec": 1152.0, "commsplit_shared_sec": 18.5},
    ]

    summary = {
        "benchmark": "nccl_commsplit_5d_parallel_sub_group_init_eager_memory_explosion_selection",
        "provenance": "synthetic + anonymized_reported (independently_reproduced=false, physical_gpu_cluster_measurement=not_measured)",
        "cluster_gpus": 4096,
        "parallel_5d_config": "TP=8, CP=2, PP=8, EP=8, DP=32",
        "naive_init_sec": 1152.0,
        "optimized_init_sec": 18.5,
        "init_speedup_x": round(1152.0 / 18.5, 1),
        "naive_nccl_hbm_gb": 14.80,
        "optimized_nccl_hbm_gb": 1.85,
        "hbm_saved_gb": round(14.80 - 1.85, 2),
        "tiers": tiers,
        "scale_points": scale_points,
    }
    return summary


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    results = evaluate_nccl_commsplit_tiers()
    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
