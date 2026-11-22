from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_results() -> dict:
    tiers = [
        {
            "tier": "Tier 1: Siloed Training & Inference Clusters + Synchronous NFS/Ceph Checkpoint (128 GPUs Total)",
            "fleet_24h_effective_utilization_pct": 38.5,
            "checkpoint_stall_9tb_sec": 225.0,
            "tidal_role_switch_sec": 540.0,
            "annual_total_cluster_tco_usd": 4560000,
        },
        {
            "tier": "Tier 2: Shared K8s Cluster (Manual Pod Restart Switching + Dedicated Lustre Storage)",
            "fleet_24h_effective_utilization_pct": 64.0,
            "checkpoint_stall_9tb_sec": 42.0,
            "tidal_role_switch_sec": 95.0,
            "annual_total_cluster_tco_usd": 3180000,
        },
        {
            "tier": "Tier 3: Kueue TAS Preemption + Two-Stage Async Host-DRAM Checkpoint + WEKA/Lustre GDS",
            "fleet_24h_effective_utilization_pct": 82.5,
            "checkpoint_stall_9tb_sec": 3.1,
            "tidal_role_switch_sec": 18.5,
            "annual_total_cluster_tco_usd": 2340000,
        },
        {
            "tier": "Tier 4: Converged Tidal AI Factory (Daytime Elastic PD + Nighttime veRL/FSDP2 Backfill + 2.8s Sleep Switch + Hyperconverged 3FS GDS + 2-Stage Async Ckpt)",
            "fleet_24h_effective_utilization_pct": 93.4,
            "checkpoint_stall_9tb_sec": 2.1,
            "tidal_role_switch_sec": 2.8,
            "annual_total_cluster_tco_usd": 1680000,
        },
    ]
    return {
        "experiment_id": "enterprise_converged_training_inference_tidal_cluster_storage_gds_tco_selection",
        "Provenance": "synthetic + anonymized_reported (independently_reproduced=false)",
        "converged_storage_and_tidal_math": {
            "two_stage_checkpoint_d2h_sec": 2.1,
            "hyperconverged_3fs_aggregate_bw_tbps": 2.34,
            "sleep_wakeup_tidal_switch_sec": 2.8,
        },
        "tiers": tiers,
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
        print("OK: enterprise_converged_training_inference_tidal_cluster_storage_gds_tco_selection verified.")
        return

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Wrote results.json")


if __name__ == "__main__":
    main()
