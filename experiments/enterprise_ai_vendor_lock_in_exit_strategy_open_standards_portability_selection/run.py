from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_payload() -> dict:
    return {
        "benchmark_id": "enterprise_ai_vendor_lock_in_exit_strategy_open_standards_portability_selection",
        "title": "Enterprise AI Anti-Vendor Lock-In & 4-Layer Open-Standards Portability Architecture Benchmark",
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload_assumptions": {
            "enterprise_ai_microservices_count": 40,
            "enterprise_mcp_tools_count": 120,
            "knowledge_base_chunks_millions": 50,
            "target_provider_switch_sla_days": 5,
        },
        "tiers": [
            {
                "tier_id": "tier1_proprietary_stateful_sdk_lock_in",
                "name": "Tier 1: Proprietary Stateful SDKs (Vendor Threads/Runs API + Closed Vector Store)",
                "provider_migration_lead_time_days": 112.0,
                "app_code_rewrite_fraction_pct": 92.0,
                "native_frontier_feature_retention_pct": 100.0,
                "provider_switch_total_cost_usd": 480000,
                "verdict": "Anti-Pattern: 16-week rewrite across 40 apps & full 50M-chunk re-parsing to switch providers",
            },
            {
                "tier_id": "tier2_lowest_common_denominator_wrapper",
                "name": "Tier 2: Naive In-Process Lowest-Common-Denominator String Wrapper",
                "provider_migration_lead_time_days": 45.0,
                "app_code_rewrite_fraction_pct": 35.0,
                "native_frontier_feature_retention_pct": 62.0,
                "provider_switch_total_cost_usd": 265000,
                "verdict": "Strips 38% of native features (loses prompt caching discounts, strict JSON & reasoning tiers)",
            },
            {
                "tier_id": "tier3_full_dual_stack_redundant_clones",
                "name": "Tier 3: Permanent Dual-Stack Redundant Prompts & Dual Closed Vector DBs",
                "provider_migration_lead_time_days": 7.0,
                "app_code_rewrite_fraction_pct": 10.0,
                "native_frontier_feature_retention_pct": 94.0,
                "provider_switch_total_cost_usd": 390000,
                "verdict": "Doubles recurring vector storage & prompt engineering maintenance ($390K/yr overhead)",
            },
            {
                "tier_id": "tier4_four_layer_open_standards_portability",
                "name": "Tier 4: 4-Layer Open Standards (Stateless Gateway + MCP Tools + Iceberg Chunks + DSPy)",
                "provider_migration_lead_time_days": 2.5,
                "app_code_rewrite_fraction_pct": 0.0,
                "native_frontier_feature_retention_pct": 99.0,
                "provider_switch_total_cost_usd": 18000,
                "verdict": "Recommended: 2.5-day switch, 0.0% app code rewrite, 99.0% feature retention & -96.3% exit cost",
            },
        ],
        "headline_findings": {
            "migration_time_reduction_days": 109.5,
            "app_code_rewrite_pct": 0.0,
            "native_feature_retention_pct": 99.0,
            "exit_cost_savings_vs_tier1_pct": 96.3,
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
