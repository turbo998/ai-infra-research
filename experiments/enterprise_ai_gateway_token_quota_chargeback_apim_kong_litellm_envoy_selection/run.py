from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_payload() -> dict:
    return {
        "experiment_id": "enterprise_ai_gateway_token_quota_chargeback_apim_kong_litellm_envoy_selection",
        "title": "Enterprise AI Gateway Token Quota Governance, Multi-BU FinOps Chargeback & Guardrail Selection Guide: Azure APIM vs. Kong AI Gateway vs. LiteLLM Enterprise vs. Envoy AI Gateway",
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload_basis": {
            "business_units_count": 10,
            "shared_base_pool": "Shared PTU + Self-Hosted K8s GPU Pool with Per-BU Spillover Subscriptions",
        },
        "tiers": [
            {
                "tier": "Tier 1: Legacy HTTP RPM Rate-Limiting + Equal Request-Count Chargeback",
                "chargeback_error_rate_pct": 38.5,
                "shared_pool_effective_utilization_pct": 48.0,
                "noisy_neighbor_collateral_drop_pct": 29.4,
                "monthly_net_spend_usd": 145000,
                "recommended_scenario": "Fails completely on LLM traffic where a single 64K+CoT call equals 420 short requests",
            },
            {
                "tier": "Tier 2: Static Hard-Partitioned TPM Quotas + Single-Phase max_tokens Pre-Deduction",
                "chargeback_error_rate_pct": 16.2,
                "shared_pool_effective_utilization_pct": 52.5,
                "noisy_neighbor_collateral_drop_pct": 1.2,
                "monthly_net_spend_usd": 118000,
                "recommended_scenario": "Strands 47.5% of shared capacity due to non-borrowable static slices and phantom max_tokens locks",
            },
            {
                "tier": "Tier 3: Two-Phase Streaming Token Bucket (P80 Soft Reserve + Post-SSE Reconcile) + USD Hard Caps",
                "chargeback_error_rate_pct": 0.8,
                "shared_pool_effective_utilization_pct": 76.0,
                "noisy_neighbor_collateral_drop_pct": 0.3,
                "monthly_net_spend_usd": 78000,
                "recommended_scenario": "Solid departmental budget governance using LiteLLM Enterprise or Azure APIM token policies",
            },
            {
                "tier": "Tier 4 (Recommended): Two-Phase 4-Tuple Weighted Token Bucket + Elastic Idle Quota Borrowing + Per-BU PAYGO Spillover",
                "chargeback_error_rate_pct": 0.02,
                "shared_pool_effective_utilization_pct": 89.4,
                "noisy_neighbor_collateral_drop_pct": 0.0,
                "monthly_net_spend_usd": 56500,
                "recommended_scenario": "Enterprise CCoE FinOps standard reaching 89.4% shared pool utilization and -61.0% net spend",
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
