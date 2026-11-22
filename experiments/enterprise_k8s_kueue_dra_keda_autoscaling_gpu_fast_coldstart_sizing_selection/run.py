from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_results() -> dict:
    tiers = [
        {
            "tier": "Tier 1: Vanilla K8s Device Plugin + HPA on DCGM_GPU_UTIL + Serial Safetensors Load (Anti-Pattern)",
            "replica_cold_start_sec": 540.0,
            "gang_deadlock_or_pcie_split_pct": 38.5,
            "cluster_24h_utilization_pct": 36.0,
            "burst_sla_drop_pct": 44.2,
        },
        {
            "tier": "Tier 2: KEDA on vLLM Queue/KV Metrics + Local NVMe HostPath Cache",
            "replica_cold_start_sec": 78.0,
            "gang_deadlock_or_pcie_split_pct": 14.0,
            "cluster_24h_utilization_pct": 62.0,
            "burst_sla_drop_pct": 9.8,
        },
        {
            "tier": "Tier 3: Kueue TAS (Topology-Aware Gang Scheduling) + K8s DRA + Run:ai Model Streamer / GDS",
            "replica_cold_start_sec": 14.2,
            "gang_deadlock_or_pcie_split_pct": 0.0,
            "cluster_24h_utilization_pct": 81.5,
            "burst_sla_drop_pct": 1.4,
        },
        {
            "tier": "Tier 4: Kueue TAS + DRA + KEDA (KV/Waiting Queue) + GDS Streamer + vLLM/SGLang Level-1 Sleep Wakeup",
            "replica_cold_start_sec": 2.8,
            "gang_deadlock_or_pcie_split_pct": 0.0,
            "cluster_24h_utilization_pct": 91.2,
            "burst_sla_drop_pct": 0.2,
        },
    ]
    return {
        "experiment_id": "enterprise_k8s_kueue_dra_keda_autoscaling_gpu_fast_coldstart_sizing_selection",
        "Provenance": "synthetic + anonymized_reported (independently_reproduced=false)",
        "coldstart_breakdown_140gb_weights": {
            "serial_http_pull_plus_disk_sec": 540.0,
            "local_nvme_mmap_sec": 78.0,
            "runai_streamer_or_gds_parallel_sec": 14.2,
            "sleep_mode_host_pinned_dram_wakeup_sec": 2.8,
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
        print("OK: enterprise_k8s_kueue_dra_keda_autoscaling_gpu_fast_coldstart_sizing_selection verified.")
        return

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Wrote results.json")


if __name__ == "__main__":
    main()
