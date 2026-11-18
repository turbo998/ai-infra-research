from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_payload() -> dict[str, Any]:
    return {
        "benchmark_id": "enterprise_wealth_management_investment_research_compliance_agent_selection_v1",
        "title": "Enterprise Wealth Management, Equity Research (SEC 10-K / Earnings Call), XBRL Reconciliation & SEC/FINRA Compliance Advisory Agent Evaluator",
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload_baseline": {
            "covered_global_equities": 3500,
            "sec_10k_10q_and_earnings_transcripts": 28000,
            "monthly_institutional_research_queries": 18000,
            "sla_target": "Financial metric & ratio accuracy >= 99.0%, numerical/unit hallucination rate == 0.0%, memo turnaround <= 2.0 minutes, 100% SEC/FINRA restricted-list compliance",
        },
        "tiers": [
            {
                "tier_id": "tier1_manual_analyst_ctrl_f_excel_modeling",
                "name": "Tier 1: Manual Equity Analyst Ctrl+F over PDF Filings + Manual Excel Copy-Paste",
                "financial_metric_and_ratio_accuracy_pct": 94.2,
                "numerical_hallucination_or_unit_mismatch_rate_pct": 3.8,
                "research_memo_turnaround_minutes": 240.0,
                "monthly_research_and_compliance_cost_usd": 165000,
                "action_decision": "Send to a human",
            },
            {
                "tier_id": "tier2_naive_vector_chunk_rag_llm_mental_math",
                "name": "Tier 2: Naive 512-Token Vector RAG + LLM Mental Math for YoY Growth & Ratios",
                "financial_metric_and_ratio_accuracy_pct": 63.5,
                "numerical_hallucination_or_unit_mismatch_rate_pct": 31.4,
                "research_memo_turnaround_minutes": 1.8,
                "monthly_research_and_compliance_cost_usd": 195000,
                "action_decision": "Send to a human",
            },
            {
                "tier_id": "tier3_long_context_128k_stuffing_without_xbrl_or_sandbox",
                "name": "Tier 3: 128K Long-Context Filing Stuffing without XBRL Fact Store or Python Sandbox",
                "financial_metric_and_ratio_accuracy_pct": 81.6,
                "numerical_hallucination_or_unit_mismatch_rate_pct": 11.2,
                "research_memo_turnaround_minutes": 3.2,
                "monthly_research_and_compliance_cost_usd": 88000,
                "action_decision": "Confirm",
            },
            {
                "tier_id": "tier4_xbrl_sql_store_plus_hierarchical_rag_plus_python_sandbox_sec_gate",
                "name": "Tier 4: Deterministic XBRL/SQL Fact Store + Hierarchical 10-K/Transcript RAG + Python Sandbox Math + SEC/FINRA Restricted-List Gate",
                "financial_metric_and_ratio_accuracy_pct": 99.4,
                "numerical_hallucination_or_unit_mismatch_rate_pct": 0.0,
                "research_memo_turnaround_minutes": 1.4,
                "monthly_research_and_compliance_cost_usd": 24000,
                "action_decision": "Act automatically",
            },
        ],
        "perturbation_check": {
            "scenario": "Cross-Period GAAP vs Non-GAAP Operating Margin Reconciliation (3M vs 9M YTD + Currency Restatement in Footnote 14)",
            "tier2_vector_rag_accuracy_pct": 51.0,
            "tier4_xbrl_python_sandbox_accuracy_pct": 99.2,
            "accuracy_gain_pts": 48.2,
            "tier1_manual_turnaround_minutes": 240.0,
            "tier4_agent_turnaround_minutes": 1.4,
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
