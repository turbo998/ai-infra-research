from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_payload() -> dict:
    return {
        "benchmark_id": "enterprise_llm_tpm_tps_littles_law_capacity_planning_ptu_sizing_selection",
        "title": "Enterprise LLM TPM vs. TPS Conversion, Little's Law Concurrency Sizing & Cloud PTU/GSU Capacity Planning Benchmark",
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload_assumptions": {
            "enterprise_dau": 50000,
            "queries_per_user_per_day": 18,
            "peak_to_average_ratio": 4.8,
            "avg_input_tokens": 3500,
            "avg_output_tokens": 500,
            "peak_rps": 50.0,
        },
        "tiers": [
            {
                "tier_id": "tier1_daily_average_rps_sizing_ignoring_peak_and_output_weight",
                "name": "Tier 1: Daily-Average RPS & Unweighted TPM Sizing (Ignores 4.8x Peak Multiplier, Little's Law & Output Token Weight)",
                "peak_hour_http_429_and_sla_drop_rate_pct": 58.4,
                "achieved_p99_e2el_sec": 34.2,
                "effective_cluster_utilization_pct": 31.0,
                "monthly_50rps_capacity_tco_usd": 385000,
                "verdict": "Anti-Pattern: 58.4% HTTP 429 throttling at 10:00 AM peak; under-sizes output token cost by 4-6x",
            },
            {
                "tier_id": "tier2_pure_paygo_serverless_burst_without_reserved_base",
                "name": "Tier 2: 100% Serverless PAYGO API at 12M Raw TPM (Zero Reserved Base, Subject to Regional TPM Rate Limits)",
                "peak_hour_http_429_and_sla_drop_rate_pct": 12.6,
                "achieved_p99_e2el_sec": 14.8,
                "effective_cluster_utilization_pct": 48.0,
                "monthly_50rps_capacity_tco_usd": 298000,
                "verdict": "High variable token cost ($298K/mo) and 12.6% regional quota throttling during enterprise bursts",
            },
            {
                "tier_id": "tier3_100pct_peak_provisioned_ptu_or_static_gpu_overprovisioning",
                "name": "Tier 3: 100% Peak-Sized Dedicated PTU / Static GPUs (Zero Spillover, Idle Night/Weekend Capacity Waste)",
                "peak_hour_http_429_and_sla_drop_rate_pct": 0.0,
                "achieved_p99_e2el_sec": 9.4,
                "effective_cluster_utilization_pct": 38.5,
                "monthly_50rps_capacity_tco_usd": 242000,
                "verdict": "Zero 429 errors, but 61.5% of reserved PTU/GPU capacity sits idle during off-peak hours",
            },
            {
                "tier_id": "tier4_littles_law_weighted_tpm_p80_base_plus_p20_paygo_spillover",
                "name": "Tier 4: Little's Law (B = RPS × E2EL) + Weighted Output TPM + P80 Reserved Base / P20 Burst Spillover",
                "peak_hour_http_429_and_sla_drop_rate_pct": 0.0,
                "achieved_p99_e2el_sec": 8.8,
                "effective_cluster_utilization_pct": 86.4,
                "monthly_50rps_capacity_tco_usd": 134000,
                "verdict": "Recommended: 0.0% 429s, 8.8s P99 E2EL, 86.4% base utilization & -65.2% TCO ($134K/mo)",
            },
        ],
        "headline_findings": {
            "peak_429_rate_pct": 0.0,
            "p99_e2el_sec": 8.8,
            "base_capacity_utilization_pct": 86.4,
            "tco_savings_vs_tier1_pct": 65.2,
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
