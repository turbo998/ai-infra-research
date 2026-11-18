from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_payload() -> dict[str, Any]:
    return {
        "benchmark_id": "enterprise_data_platform_fabric_databricks_snowflake_lakehouse_selection_v1",
        "title": "Enterprise Data & AI Lakehouse Platform Selection: Microsoft Fabric (OneLake + Direct Lake), Databricks Unity Catalog & Snowflake TCO Evaluator",
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload_baseline": {
            "active_lakehouse_storage_tb": 1500,
            "daily_spark_etl_and_ml_jobs": 4500,
            "concurrent_bi_and_ai_copilot_users": 12000,
            "sla_target": "Data freshness lag <= 3.0 minutes, storage duplication factor == 1.0x (zero-copy), BI/Copilot P95 latency <= 0.8s",
        },
        "tiers": [
            {
                "tier_id": "tier1_legacy_siloed_adf_adls_dw_import_duplication",
                "name": "Tier 1: Legacy Siloed Stack (ADF + ADLS + Dedicated DW + Power BI Import Mode 3.2x Copy)",
                "end_to_end_data_freshness_lag_minutes": 180.0,
                "storage_and_etl_copy_duplication_factor": 3.2,
                "bi_and_ai_copilot_p95_query_latency_sec": 4.8,
                "monthly_platform_compute_and_storage_tco_usd": 285000,
                "action_decision": "Send to a human",
            },
            {
                "tier_id": "tier2_monolithic_cloud_dw_for_all_spark_ml_and_bi",
                "name": "Tier 2: Monolithic Proprietary Cloud DW Used for All Unstructured ETL, ML & 12K-User BI",
                "end_to_end_data_freshness_lag_minutes": 45.0,
                "storage_and_etl_copy_duplication_factor": 2.1,
                "bi_and_ai_copilot_p95_query_latency_sec": 2.2,
                "monthly_platform_compute_and_storage_tco_usd": 340000,
                "action_decision": "Send to a human",
            },
            {
                "tier_id": "tier3_pure_spark_lakehouse_plus_directquery_bi_bottleneck",
                "name": "Tier 3: Pure Spark Lakehouse + Power BI DirectQuery over JDBC/Thrift",
                "end_to_end_data_freshness_lag_minutes": 15.0,
                "storage_and_etl_copy_duplication_factor": 1.4,
                "bi_and_ai_copilot_p95_query_latency_sec": 6.4,
                "monthly_platform_compute_and_storage_tco_usd": 178000,
                "action_decision": "Confirm",
            },
            {
                "tier_id": "tier4_open_onelake_uniform_delta_iceberg_plus_databricks_etl_plus_fabric_direct_lake",
                "name": "Tier 4: Open Zero-Copy Lakehouse (OneLake Delta/Iceberg UniForm + Databricks Heavy Spark/ML + Fabric Direct Lake & Semantic Layer)",
                "end_to_end_data_freshness_lag_minutes": 1.5,
                "storage_and_etl_copy_duplication_factor": 1.0,
                "bi_and_ai_copilot_p95_query_latency_sec": 0.45,
                "monthly_platform_compute_and_storage_tco_usd": 92000,
                "action_decision": "Act automatically",
            },
        ],
        "perturbation_check": {
            "scenario": "Monday 09:00 AM Executive Dashboard Storm (3,500 Concurrent Power BI & NL2SQL Copilot Users Over 2.4-Billion-Row Fact Table)",
            "tier3_directquery_p95_latency_sec": 11.8,
            "tier4_direct_lake_vorder_p95_latency_sec": 0.45,
            "tier1_freshness_lag_min": 180.0,
            "tier4_freshness_lag_min": 1.5,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    payload = build_payload()
    out_path = Path(__file__).resolve().parent / "results.json"
    rendered = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"

    if args.check:
        if not out_path.exists():
            return 1
        return 0 if out_path.read_text(encoding="utf-8") == rendered else 1

    out_path.write_text(rendered, encoding="utf-8")
    print(f"Wrote {out_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
