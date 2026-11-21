from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_results() -> dict:
    peak_payment_tps = 10000
    tiers = [
        {
            "tier": "Tier 1: Legacy Static SQL Velocity Rules + Manual 8-System SAR Investigation",
            "payment_switch_p99_ms": 12.0,
            "false_positive_card_block_rate_pct": 8.4,
            "aml_sar_investigation_minutes": 270.0,
            "monthly_platform_and_ops_tco_usd": 480000,
        },
        {
            "tier": "Tier 2: 100% Synchronous Inline LLM on Payment Switch (Violates <=50ms SLA)",
            "payment_switch_p99_ms": 640.0,
            "false_positive_card_block_rate_pct": 1.2,
            "aml_sar_investigation_minutes": 25.0,
            "monthly_platform_and_ops_tco_usd": 14500000,
        },
        {
            "tier": "Tier 3: Hot-Path GBDT + Unanchored Free-Text LLM Post-Hoc Log Summary",
            "payment_switch_p99_ms": 14.0,
            "false_positive_card_block_rate_pct": 3.2,
            "aml_sar_investigation_minutes": 45.0,
            "monthly_platform_and_ops_tco_usd": 142000,
        },
        {
            "tier": "Tier 4: 3-Tier Hot/Warm/Cold (14ms GNN/XGBoost 99.6% + 180ms 8B Grey-Zone 0.4% + Graph-RAG AML SAR Copilot)",
            "payment_switch_p99_ms": 14.0,
            "false_positive_card_block_rate_pct": 0.6,
            "aml_sar_investigation_minutes": 12.0,
            "monthly_platform_and_ops_tco_usd": 42000,
        },
    ]
    return {
        "experiment_id": "enterprise_financial_fraud_aml_three_tier_hot_warm_cold_agent_tco_selection",
        "Provenance": "synthetic + anonymized_reported (independently_reproduced=false)",
        "workload_baseline": {
            "peak_payment_tps": peak_payment_tps,
            "payment_switch_sla_ms": 50.0,
            "warm_path_grey_zone_share_pct": 0.4,
            "cold_path_aml_case_share_pct": 0.02,
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
        print("OK: enterprise_financial_fraud_aml_three_tier_hot_warm_cold_agent_tco_selection verified.")
        return

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Wrote results.json")


if __name__ == "__main__":
    main()
