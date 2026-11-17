from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_Payload() -> dict:
    return {
        "round": 405,
        "slug": "enterprise_financial_fraud_aml_agent_selection",
        "title": "Enterprise Financial Fraud Detection, AML Investigation & Real-Time Transaction Risk Agent Architecture Selection",
        "Provenance": "synthetic_analytical_model_calibrated_to_published_payment_risk_gnn_xgboost_fincen_sar_aml_agent_and_tiered_latency_benchmarks",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload_assumptions": {
            "monthly_transactions": 100000000,
            "peak_tps": 3000,
            "monthly_aml_alerts": 120000,
            "enterprise_sla": "P99 synchronous payment authorization <= 50ms, false positive alert rate (FPR) <= 20%, AML SAR investigation <= 5 min/case",
        },
        "tiers": [
            {
                "tier_id": "tier1_legacy_sql_velocity_rules_manual_aml",
                "name": "Tier 1: Legacy Static SQL Velocity Rules + Manual 8-System AML Investigation",
                "false_positive_alert_rate_pct": 94.2,
                "payment_auth_p99_latency_ms": 12.0,
                "aml_case_review_minutes": 45.0,
                "monthly_total_cost_usd": 185000,
                "action": "Send to a human",
                "status": "fail",
            },
            {
                "tier_id": "tier2_sync_inline_llm_on_payment_switch",
                "name": "Tier 2: Synchronous Inline LLM in Real-Time Payment Authorization Path",
                "false_positive_alert_rate_pct": 38.0,
                "payment_auth_p99_latency_ms": 1450.0,
                "aml_case_review_minutes": 15.0,
                "monthly_total_cost_usd": 420000,
                "action": "Send to a human",
                "status": "fail",
            },
            {
                "tier_id": "tier3_inline_xgboost_plus_unstructured_llm_summary",
                "name": "Tier 3: Inline XGBoost (<15ms) + Unstructured Free-Text LLM Log Summary",
                "false_positive_alert_rate_pct": 42.5,
                "payment_auth_p99_latency_ms": 14.0,
                "aml_case_review_minutes": 12.0,
                "monthly_total_cost_usd": 68000,
                "action": "Confirm",
                "status": "warn",
            },
            {
                "tier_id": "tier4_hot_gnn_xgboost_warm_8b_greyzone_cold_graph_aml_copilot",
                "name": "Tier 4: Hot-Path 14ms GNN/XGBoost + Warm-Path 0.4% Grey-Zone 8B Evaluator + Cold-Path Graph-RAG FinCEN SAR Copilot",
                "false_positive_alert_rate_pct": 16.4,
                "payment_auth_p99_latency_ms": 14.0,
                "aml_case_review_minutes": 3.2,
                "monthly_total_cost_usd": 24500,
                "action": "Act automatically",
                "status": "pass",
            },
        ],
        "robustness_check": [
            {
                "metric": "False Positive Alert Rate FPR (%)",
                "baseline_tier1": 94.2,
                "optimized_tier4": 16.4,
                "delta": "-77.8 pts (-82.6% false alerts)",
            },
            {
                "metric": "Payment Auth P99 Latency (ms)",
                "baseline_tier2": 1450.0,
                "optimized_tier4": 14.0,
                "delta": "103.6x faster (Meets <50ms SLA)",
            },
            {
                "metric": "AML SAR Case Review Time (Min)",
                "baseline_tier1": 45.0,
                "optimized_tier4": 3.2,
                "delta": "14.1x faster analyst review",
            },
            {
                "metric": "Monthly Total Risk Ops Cost (USD)",
                "baseline_tier1": 185000,
                "optimized_tier4": 24500,
                "delta": "-86.8% (-$160,500/mo)",
            },
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    out_path = Path(__file__).with_name("results.json")
    payload = build_Payload()

    if args.check:
        if not out_path.exists():
            raise SystemExit("results.json missing")
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        if existing != payload:
            raise SystemExit("results.json out of sync")
        print("OK: enterprise_financial_fraud_aml_agent_selection verified.")
        return 0

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
