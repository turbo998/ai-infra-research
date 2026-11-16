from __future__ import annotations

import json
from pathlib import Path


def evaluate_triton_jit_cache_tiers() -> dict:
    tiers = [
        {
            "tier": "Tier 1: Default Shared NFS ~/.triton/cache Across 512 Ranks",
            "cold_startup_latency_sec": 840.0,
            "corrupt_cubin_crash_rate_pct": 38.5,
            "node_tmp_disk_gb": 1.80,
            "ptxas_cpu_redundancy_x": 512,
            "decision": "Send to a human",
        },
        {
            "tier": "Tier 2: Per-Rank Isolated /tmp/triton_rank_${LOCAL_RANK}",
            "cold_startup_latency_sec": 168.0,
            "corrupt_cubin_crash_rate_pct": 0.0,
            "node_tmp_disk_gb": 14.40,
            "ptxas_cpu_redundancy_x": 8,
            "decision": "Confirm",
        },
        {
            "tier": "Tier 3: Shared Node-Local /tmp/triton Without Local-Rank-0 Barrier",
            "cold_startup_latency_sec": 142.0,
            "corrupt_cubin_crash_rate_pct": 4.2,
            "node_tmp_disk_gb": 1.80,
            "ptxas_cpu_redundancy_x": 8,
            "decision": "Send to a human",
        },
        {
            "tier": "Tier 4: Node-Local /tmp/triton + LOCAL_RANK==0 Warmup Barrier + Atomic Rename",
            "cold_startup_latency_sec": 24.9,
            "corrupt_cubin_crash_rate_pct": 0.0,
            "node_tmp_disk_gb": 1.80,
            "ptxas_cpu_redundancy_x": 1,
            "decision": "Act automatically",
        },
    ]

    scale_points = [
        {"gpus": 64, "nfs_shared_sec": 145.0, "per_rank_tmp_sec": 165.0, "rank0_local_sec": 24.5},
        {"gpus": 128, "nfs_shared_sec": 260.0, "per_rank_tmp_sec": 166.0, "rank0_local_sec": 24.6},
        {"gpus": 256, "nfs_shared_sec": 480.0, "per_rank_tmp_sec": 167.0, "rank0_local_sec": 24.7},
        {"gpus": 512, "nfs_shared_sec": 840.0, "per_rank_tmp_sec": 168.0, "rank0_local_sec": 24.9},
    ]

    summary = {
        "benchmark": "triton_jit_cache_nfs_race_corrupt_cubin_local_shm_rank0_warmup_selection",
        "provenance": "synthetic + anonymized_reported (independently_reproduced=false, physical_gpu_cluster_measurement=not_measured)",
        "cluster_gpus": 512,
        "gpus_per_node": 8,
        "nfs_cold_startup_sec": 840.0,
        "per_rank_cold_startup_sec": 168.0,
        "rank0_barrier_cold_startup_sec": 24.9,
        "speedup_vs_nfs_x": round(840.0 / 24.9, 1),
        "speedup_vs_per_rank_x": round(168.0 / 24.9, 1),
        "nfs_crash_rate_pct": 38.5,
        "tiers": tiers,
        "scale_points": scale_points,
    }
    return summary


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    results = evaluate_triton_jit_cache_tiers()
    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
