from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_payload() -> dict:
    return {
        "benchmark_id": "enterprise_b2b_rfp_security_questionnaire_sig_caiq_automation_agent_selection",
        "title": "Enterprise B2B RFP & Security Questionnaire (SIG/CAIQ/SOC2) Automation Agent Benchmark",
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload_assumptions": {
            "monthly_rfps_and_questionnaires": 800,
            "avg_questions_per_rfp": 250,
            "canonical_approved_qa_pairs": 2500,
            "sme_attestation_ttl_days": 90,
        },
        "tiers": [
            {
                "tier_id": "tier1_uncurated_wiki_and_past_proposals_rag",
                "name": "Tier 1: Naive RAG Over Uncurated Engineering Wikis & Past Proposals",
                "binding_over_commitment_error_rate_pct": 21.5,
                "verified_auto_completion_rate_pct": 62.0,
                "rfp_turnaround_hours": 72.0,
                "monthly_presales_and_liability_tco_usd": 240000,
                "verdict": "Anti-Pattern: Promises internal Q4 roadmap ideas & one-off custom deals as standard GA features (21.5% liability!)",
            },
            {
                "tier_id": "tier2_legacy_keyword_qa_bank_manual_copy_paste",
                "name": "Tier 2: Legacy Keyword Q&A Spreadsheet Search + Manual SE Copy-Paste",
                "binding_over_commitment_error_rate_pct": 3.5,
                "verified_auto_completion_rate_pct": 48.0,
                "rfp_turnaround_hours": 228.0,
                "monthly_presales_and_liability_tco_usd": 195000,
                "verdict": "9.5-day (228h) turnaround; 52% of compound questions escalated to busy SEs & Security SMEs",
            },
            {
                "tier_id": "tier3_uncalibrated_100pct_autofill_no_confidence_bands",
                "name": "Tier 3: 100% Forced Auto-Fill Without Confidence Bands or Dropdown Constraints",
                "binding_over_commitment_error_rate_pct": 12.8,
                "verified_auto_completion_rate_pct": 71.0,
                "rfp_turnaround_hours": 48.0,
                "monthly_presales_and_liability_tco_usd": 165000,
                "verdict": "Hallucinates low-confidence answers & breaks customer Excel data-validation dropdowns",
            },
            {
                "tier_id": "tier4_canonical_qa_bank_ga_firewall_3band_router_excel_writeback",
                "name": "Tier 4: Canonical 90d Q&A Bank + GA-Only Firewall + 3-Band Confidence Router + Excel Dropdown Writeback",
                "binding_over_commitment_error_rate_pct": 0.0,
                "verified_auto_completion_rate_pct": 91.5,
                "rfp_turnaround_hours": 0.63,
                "monthly_presales_and_liability_tco_usd": 28000,
                "verdict": "Recommended: 0.0% roadmap over-commitment, 91.5% verified completion, 38-min turnaround & -88.3% TCO",
            },
        ],
        "headline_findings": {
            "over_commitment_liability_rate_pct": 0.0,
            "verified_auto_completion_pct": 91.5,
            "turnaround_time_minutes": 38.0,
            "tco_savings_vs_tier1_pct": 88.3,
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
