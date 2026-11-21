from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_results() -> dict:
    monthly_reports = 100000
    tiers = [
        {
            "tier": "Tier 1: Monolithic Serial 40-Step 70B/R1 ReAct Loop (All Docs Stuffed in One Context)",
            "tokens_per_report": 1240000,
            "wall_clock_seconds": 680,
            "numerical_audit_accuracy_pct": 81.5,
            "cost_per_report_usd": 2.48,
            "monthly_tco_usd": 248000,
        },
        {
            "tier": "Tier 2: Commercial Public Web Deep Research API (OpenAI / Perplexity Deep Research)",
            "tokens_per_report": 420000,
            "wall_clock_seconds": 240,
            "numerical_audit_accuracy_pct": 91.0,
            "cost_per_report_usd": 1.65,
            "monthly_tco_usd": 165000,
        },
        {
            "tier": "Tier 3: Static Multi-Agent Pipeline (No Python Accounting Identity Sandbox)",
            "tokens_per_report": 310000,
            "wall_clock_seconds": 165,
            "numerical_audit_accuracy_pct": 88.4,
            "cost_per_report_usd": 0.62,
            "monthly_tco_usd": 62000,
        },
        {
            "tier": "Tier 4: Parallel Map-Reduce Sub-Agent DAG (8x 8B/32B Workers) + Python Accounting Sandbox + Cell Citation Gate",
            "tokens_per_report": 230000,
            "wall_clock_seconds": 110,
            "numerical_audit_accuracy_pct": 99.4,
            "cost_per_report_usd": 0.38,
            "monthly_tco_usd": 38000,
        },
    ]
    return {
        "experiment_id": "enterprise_financial_deep_research_due_diligence_agent_dag_citation_tco_selection",
        "Provenance": "synthetic + anonymized_reported (independently_reproduced=false)",
        "workload_baseline": {
            "monthly_research_reports": monthly_reports,
            "sources_inspected_per_report": 40,
            "parallel_map_workers": 8,
            "max_evidence_card_tokens": 600,
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
        print("OK: enterprise_financial_deep_research_due_diligence_agent_dag_citation_tco_selection verified.")
        return

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Wrote results.json")


if __name__ == "__main__":
    main()
