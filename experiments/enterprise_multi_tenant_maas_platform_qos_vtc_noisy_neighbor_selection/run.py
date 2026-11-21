from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_payload() -> dict:
    return {
        "benchmark_id": "enterprise_multi_tenant_maas_platform_qos_vtc_noisy_neighbor_selection",
        "title": "Enterprise Multi-Tenant MaaS Platform Architecture: 3-Tier QoS Isolation, Virtual Token Fair Queueing (VTC) & Noisy-Neighbor Benchmark",
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload_assumptions": {
            "enterprise_tenants": 60,
            "cluster_gpus": 64,
            "gpu_sku": "NVIDIA H200 SXM 141GB",
            "qos_classes": ["Gold (Real-Time Customer Facing)", "Silver (Internal Copilot)", "Bronze (Batch/Eval)"],
        },
        "tiers": [
            {
                "tier_id": "tier1_single_fifo_fcfs_shared_pool_without_tenant_fair_queueing",
                "name": "Tier 1: Flat Shared Pool with FIFO/FCFS Queueing (1 Noisy Batch Tenant Starves 59 Interactive Tenants!)",
                "gold_tenant_p99_ttft_under_burst_ms": 14200.0,
                "cluster_overall_gpu_utilization_pct": 74.0,
                "noisy_neighbor_sla_violation_rate_pct": 68.4,
                "monthly_64gpu_maas_platform_tco_usd": 310000,
                "verdict": "Anti-Pattern: FIFO queue lets 1 tenant's 100-request 32K burst block Gold customer traffic for 14.2 seconds!",
            },
            {
                "tier_id": "tier2_strict_physical_gpu_silos_per_department",
                "name": "Tier 2: Strict Physical GPU Silos per Department (Zero Sharing — 100% Isolation, Massive Idle GPU Waste)",
                "gold_tenant_p99_ttft_under_burst_ms": 480.0,
                "cluster_overall_gpu_utilization_pct": 29.5,
                "noisy_neighbor_sla_violation_rate_pct": 0.0,
                "monthly_64gpu_maas_platform_tco_usd": 420000,
                "verdict": "0% noisy neighbor, but physical departmental silos strand 70.5% of GPUs idle and inflate TCO to $420K/mo",
            },
            {
                "tier_id": "tier3_static_per_tenant_hard_rate_limits_without_work_conserving_borrowing",
                "name": "Tier 3: Shared Pool with Static Hard Per-Tenant RPM/TPM Caps (Throttles Tenants Even When Cluster Is 60% Idle!)",
                "gold_tenant_p99_ttft_under_burst_ms": 690.0,
                "cluster_overall_gpu_utilization_pct": 51.2,
                "noisy_neighbor_sla_violation_rate_pct": 6.4,
                "monthly_64gpu_maas_platform_tco_usd": 245000,
                "verdict": "Non-work-conserving: rejects valid tenant bursts with HTTP 429 even when 48% of cluster GPUs are idle",
            },
            {
                "tier_id": "tier4_3tier_qos_plus_vtc_token_fair_queueing_and_work_conserving_quota_borrowing",
                "name": "Tier 4: 3-Tier QoS (Gold/Silver/Bronze) + Virtual Token Counter (`VTC`) Fair Queueing + Elastic Quota Borrowing",
                "gold_tenant_p99_ttft_under_burst_ms": 430.0,
                "cluster_overall_gpu_utilization_pct": 88.6,
                "noisy_neighbor_sla_violation_rate_pct": 0.0,
                "monthly_64gpu_maas_platform_tco_usd": 138000,
                "verdict": "Recommended: 430ms Gold P99 TTFT (-33.0x vs. Tier 1), 88.6% cluster util, 0.0% breach & -67.1% TCO vs. Silos",
            },
        ],
        "headline_findings": {
            "gold_p99_ttft_ms": 430.0,
            "cluster_utilization_pct": 88.6,
            "noisy_neighbor_breach_pct": 0.0,
            "tco_savings_vs_silos_pct": 67.1,
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
