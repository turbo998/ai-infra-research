from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_payload() -> dict[str, Any]:
    return {
        "benchmark_id": "enterprise_customs_tariff_hs_code_export_control_agent_selection_v1",
        "title": "Enterprise Global Trade & Customs Compliance: 10-Digit HS Code GRI Tree, FTA Rules-of-Origin RVC & OFAC/BIS Export Control Agent Evaluator",
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload_baseline": {
            "monthly_customs_declarations": 95000,
            "active_engineering_and_retail_skus": 320000,
            "global_trading_counterparties": 55000,
            "sla_target": "10-digit HS classification accuracy >= 95%, OFAC/BIS export-control miss rate == 0.0%, declaration prep <= 3.0 minutes",
        },
        "tiers": [
            {
                "tier_id": "tier1_manual_customs_broker_and_spreadsheet_roo",
                "name": "Tier 1: Manual Customs Broker Tariff Lookup + Spreadsheet FTA Origin Calculation",
                "ten_digit_hs_classification_accuracy_pct": 89.0,
                "sanctions_or_eccn_compliance_miss_rate_pct": 1.6,
                "declaration_clearance_prep_minutes": 65.0,
                "monthly_brokerage_and_penalty_cost_usd": 275000,
                "action_decision": "Send to a human",
            },
            {
                "tier_id": "tier2_prompt_only_llm_freeform_hs_code_and_screening",
                "name": "Tier 2: Prompt-Only LLM Generating 10-Digit HS Codes & Judging Export Controls Directly",
                "ten_digit_hs_classification_accuracy_pct": 62.5,
                "sanctions_or_eccn_compliance_miss_rate_pct": 12.4,
                "declaration_clearance_prep_minutes": 2.2,
                "monthly_brokerage_and_penalty_cost_usd": 410000,
                "action_decision": "Send to a human",
            },
            {
                "tier_id": "tier3_flat_vector_hs_table_rag_plus_keyword_ofac",
                "name": "Tier 3: Flat Vector Search over HS Descriptions + Exact Keyword Sanction Screening",
                "ten_digit_hs_classification_accuracy_pct": 76.8,
                "sanctions_or_eccn_compliance_miss_rate_pct": 4.5,
                "declaration_clearance_prep_minutes": 14.0,
                "monthly_brokerage_and_penalty_cost_usd": 132000,
                "action_decision": "Confirm",
            },
            {
                "tier_id": "tier4_wco_gri_chapter_note_tree_plus_bom_rvc_engine_plus_ofac_50pct_graph_agent",
                "name": "Tier 4: Hierarchical WCO GRI 1-6 & Chapter-Note Tree + Deterministic BOM RVC Calculator + OFAC 50%-Rule Ownership Graph Agent",
                "ten_digit_hs_classification_accuracy_pct": 96.6,
                "sanctions_or_eccn_compliance_miss_rate_pct": 0.0,
                "declaration_clearance_prep_minutes": 1.8,
                "monthly_brokerage_and_penalty_cost_usd": 39000,
                "action_decision": "Act automatically",
            },
        ],
        "perturbation_check": {
            "scenario": "Composite Mechatronic Sub-Assembly with Chapter 84/85 Exclusion Note + Transliterated 55%-Owned Subsidiary of OFAC SDN Entity",
            "tier2_hs_accuracy_pct": 48.5,
            "tier4_gri_tree_hs_accuracy_pct": 95.8,
            "hs_accuracy_gain_pts": 47.3,
            "tier3_keyword_ofac_miss_pct": 18.0,
            "tier4_ownership_graph_ofac_miss_pct": 0.0,
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
