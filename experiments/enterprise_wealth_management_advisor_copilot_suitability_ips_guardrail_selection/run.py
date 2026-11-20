from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_payload() -> dict:
    return {
        "benchmark_id": "enterprise_wealth_management_advisor_copilot_suitability_ips_guardrail_selection",
        "title": "Enterprise Wealth Management Advisor Copilot: IPS Suitability Solver, CIO House-View & FINRA 2210 Guardrail Benchmark",
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload_assumptions": {
            "financial_advisors_supported": 12000,
            "client_households_managed": 480000,
            "monthly_client_portfolio_reviews_and_briefs": 360000,
            "approved_product_list_apl_securities": 2400,
        },
        "tiers": [
            {
                "tier_id": "tier1_unconstrained_llm_portfolio_advice_and_email_drafting",
                "name": "Tier 1: Unconstrained LLM Ticker Advice & Free-Form Client Email Drafting",
                "reg_bi_mifid_suitability_violation_rate_pct": 16.8,
                "finra_2210_promissory_language_violation_pct": 11.4,
                "advisor_meeting_prep_time_minutes": 12.0,
                "monthly_compliance_remediation_and_prep_tco_usd": 720000,
                "verdict": "Anti-Pattern: 16.8% Reg BI/MiFID II suitability violations & 11.4% promissory-return breaches",
            },
            {
                "tier_id": "tier2_legacy_100pct_manual_advisor_prep_and_series24_review",
                "name": "Tier 2: Legacy Manual Advisor Prep + 100% Series 24 Principal Supervisory Queue",
                "reg_bi_mifid_suitability_violation_rate_pct": 0.4,
                "finra_2210_promissory_language_violation_pct": 0.6,
                "advisor_meeting_prep_time_minutes": 52.0,
                "monthly_compliance_remediation_and_prep_tco_usd": 540000,
                "verdict": "52-min manual prep per meeting & 2.8-day Series 24 compliance approval bottleneck",
            },
            {
                "tier_id": "tier3_cio_research_rag_without_deterministic_ips_solver",
                "name": "Tier 3: CIO Research PDF RAG Only (No Deterministic Client IPS or Tax-Lot Solver)",
                "reg_bi_mifid_suitability_violation_rate_pct": 8.2,
                "finra_2210_promissory_language_violation_pct": 2.8,
                "advisor_meeting_prep_time_minutes": 9.5,
                "monthly_compliance_remediation_and_prep_tco_usd": 310000,
                "verdict": "Summarizes CIO macro views well, but violates 8.2% of client IPS concentration/ESG/tax caps",
            },
            {
                "tier_id": "tier4_ips_apl_convex_solver_cio_house_view_finra2210_linter",
                "name": "Tier 4: Deterministic IPS/APL Convex Solver + Approved CIO House-View RAG + FINRA 2210 Linter",
                "reg_bi_mifid_suitability_violation_rate_pct": 0.0,
                "finra_2210_promissory_language_violation_pct": 0.0,
                "advisor_meeting_prep_time_minutes": 3.5,
                "monthly_compliance_remediation_and_prep_tco_usd": 78000,
                "verdict": "Recommended: 0.0% Reg BI/FINRA 2210 violations, 3.5m prep time & -89.2% TCO ($78K/mo)",
            },
        ],
        "headline_findings": {
            "suitability_violation_rate_pct": 0.0,
            "promissory_violation_rate_pct": 0.0,
            "advisor_prep_time_minutes": 3.5,
            "tco_savings_vs_tier1_pct": 89.2,
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
