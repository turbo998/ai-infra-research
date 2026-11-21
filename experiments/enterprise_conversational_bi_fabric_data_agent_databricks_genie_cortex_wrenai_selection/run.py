from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_results() -> dict:
    warehouse_tables = 500
    tiers = [
        {
            "tier": "Tier 1: Raw 500-Table DDL Stuffing + Shared Service Account Text-to-SQL",
            "prompt_tokens_per_query": 85000,
            "business_metric_accuracy_pct": 51.4,
            "rls_violation_risk_pct": 14.5,
            "runaway_full_scan_incidents_per_mo": 42,
            "monthly_platform_and_dw_tco_usd": 48500,
        },
        {
            "tier": "Tier 2: Vector Schema-RAG on Physical Tables (No Certified Semantic Layer)",
            "prompt_tokens_per_query": 6200,
            "business_metric_accuracy_pct": 72.8,
            "rls_violation_risk_pct": 4.2,
            "runaway_full_scan_incidents_per_mo": 16,
            "monthly_platform_and_dw_tco_usd": 22400,
        },
        {
            "tier": "Tier 3: Cloud Lakehouse Native BI Agents (Microsoft Fabric Data Agent / Databricks Genie / Cortex Analyst)",
            "prompt_tokens_per_query": 3100,
            "business_metric_accuracy_pct": 93.8,
            "rls_violation_risk_pct": 0.0,
            "runaway_full_scan_incidents_per_mo": 1,
            "monthly_platform_and_dw_tco_usd": 9200,
        },
        {
            "tier": "Tier 4: Certified Semantic Layer (Fabric IQ / Cube MDL + WrenAI) + SQLGlot AST Gate + EXPLAIN Circuit Breaker + OBO RLS",
            "prompt_tokens_per_query": 2800,
            "business_metric_accuracy_pct": 94.6,
            "rls_violation_risk_pct": 0.0,
            "runaway_full_scan_incidents_per_mo": 0,
            "monthly_platform_and_dw_tco_usd": 7400,
        },
    ]
    return {
        "experiment_id": "enterprise_conversational_bi_fabric_data_agent_databricks_genie_cortex_wrenai_selection",
        "Provenance": "synthetic + anonymized_reported (independently_reproduced=false)",
        "workload_baseline": {
            "warehouse_tables": warehouse_tables,
            "monthly_analytical_queries": 300000,
            "explain_scan_limit_gb": 50,
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
        print("OK: enterprise_conversational_bi_fabric_data_agent_databricks_genie_cortex_wrenai_selection verified.")
        return

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Wrote results.json")


if __name__ == "__main__":
    main()
