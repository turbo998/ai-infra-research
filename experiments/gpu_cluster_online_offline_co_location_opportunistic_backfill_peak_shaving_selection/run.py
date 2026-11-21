from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_payload() -> dict:
    return {
        "experiment_id": "gpu_cluster_online_offline_co_location_opportunistic_backfill_peak_shaving_selection",
        "title": "Online-Offline GPU Co-Location & Opportunistic Backfill Selection: Priority-Preemptive KV-Watermark Backfill vs. Siloed Clusters",
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload_basis": {
            "online_peak_gpus": 48,
            "offline_daily_tokens_billions": 1.2,
            "online_p99_ttft_sla_ms": 500.0,
        },
        "tiers": [
            {
                "tier": "Tier 1: Siloed Dedicated Clusters (48 Online GPUs + 16 Offline Batch GPUs = 64 GPUs)",
                "total_gpus_required": 64,
                "cluster_24h_avg_utilization_pct": 42.5,
                "online_p99_ttft_ms": 380.0,
                "online_sla_goodput_pct": 99.6,
                "monthly_cluster_cost_usd": 203200,
                "recommended_scenario": "Air-gapped environments where online and offline workloads cannot share pods",
            },
            {
                "tier": "Tier 2: Un-Isolated Shared FIFO Queue (Online + Offline Mixed on 48 GPUs)",
                "total_gpus_required": 48,
                "cluster_24h_avg_utilization_pct": 86.0,
                "online_p99_ttft_ms": 4850.0,
                "online_sla_goodput_pct": 44.0,
                "monthly_cluster_cost_usd": 152400,
                "recommended_scenario": "Anti-pattern (offline batch requests starve online users and spike P99 TTFT by 12.7x)",
            },
            {
                "tier": "Tier 3: Coarse CronHPA Day/Night Pod Reallocation (Nightly Offline Swap on 48 GPUs)",
                "total_gpus_required": 52,
                "cluster_24h_avg_utilization_pct": 72.0,
                "online_p99_ttft_ms": 410.0,
                "online_sla_goodput_pct": 98.8,
                "monthly_cluster_cost_usd": 165100,
                "recommended_scenario": "Predictable diurnal workloads with zero daytime batch urgency",
            },
            {
                "tier": "Tier 4 (Recommended): KV-Watermark-Gated Opportunistic Backfill + vLLM Priority Preemption (48 GPUs Total)",
                "total_gpus_required": 48,
                "cluster_24h_avg_utilization_pct": 91.5,
                "online_p99_ttft_ms": 395.0,
                "online_sla_goodput_pct": 99.4,
                "monthly_cluster_cost_usd": 152400,
                "recommended_scenario": "Production AI Gateway + vLLM pool harvesting both daytime micro-valleys & night idle capacity (-$50.8k/mo)",
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
