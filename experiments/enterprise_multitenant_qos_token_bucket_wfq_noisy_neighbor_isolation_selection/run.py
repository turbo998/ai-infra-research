from __future__ import annotations

import json
from pathlib import Path


def build_results() -> dict:
    tiers = [
        {
            "tier": "Tier 1: Unthrottled FIFO / Simple RPS Limit (Noisy Neighbor)",
            "vip_p99_ttft_ms": 18400.0,
            "cluster_gpu_utilization_pct": 78.0,
            "false_429_throttle_rate_pct": 18.5,
            "effective_goodput_index": 0.42,
            "monthly_wasted_gpu_usd": 64000,
            "action": "Send to a human",
        },
        {
            "tier": "Tier 2: Rigid Static Hard TPM Quota (No Idle Borrowing)",
            "vip_p99_ttft_ms": 340.0,
            "cluster_gpu_utilization_pct": 46.5,
            "false_429_throttle_rate_pct": 34.0,
            "effective_goodput_index": 0.58,
            "monthly_wasted_gpu_usd": 92000,
            "action": "Send to a human",
        },
        {
            "tier": "Tier 3: Standard Raw-TPM Token Bucket (Blind to Cache/Output Cost)",
            "vip_p99_ttft_ms": 2850.0,
            "cluster_gpu_utilization_pct": 74.0,
            "false_429_throttle_rate_pct": 11.2,
            "effective_goodput_index": 0.76,
            "monthly_wasted_gpu_usd": 38000,
            "action": "Confirm",
        },
        {
            "tier": "Tier 4: Dual-Rate CIR/EIR WCU Token Bucket + Preemptive WFQ (Recommended)",
            "vip_p99_ttft_ms": 325.0,
            "cluster_gpu_utilization_pct": 88.5,
            "false_429_throttle_rate_pct": 1.9,
            "effective_goodput_index": 1.00,
            "monthly_wasted_gpu_usd": 8500,
            "action": "Act automatically",
        },
    ]

    return {
        "experiment_id": "enterprise_multitenant_qos_token_bucket_wfq_noisy_neighbor_isolation_selection",
        "title": "Enterprise Multi-Tenant QoS, Dual-Rate CIR/EIR WCU Token Bucket & Preemptive WFQ Selection (#493)",
        "provenance": "synthetic + anonymized_reported (independently_reproduced=false)",
        "cluster_baseline": {
            "total_gpus": 64,
            "tenants": 50,
            "wcu_formula": "WCU = 1.0 * Uncached_Input + 0.1 * Cached_Input + 4.0 * Output_Tokens",
            "vip_ttft_improvement_factor_vs_rps": round(18400.0 / 325.0, 2),
            "utilization_gain_pts_vs_hard_quota": round(88.5 - 46.5, 1),
        },
        "tiers": tiers,
    }


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    results = build_results()
    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("Wrote results.json")


if __name__ == "__main__":
    main()
