from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_payload() -> dict[str, Any]:
    return {
        "benchmark_id": "enterprise_hr_talent_recruitment_policy_copilot_bias_audit_selection_v1",
        "title": "Enterprise HR Talent Acquisition, ESCO/O*NET Skill Ontology Matching, EEOC/EU AI Act Bias Audit & RBAC Employee Policy Copilot Evaluator",
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload_baseline": {
            "global_employees": 45000,
            "monthly_candidate_resumes": 180000,
            "monthly_employee_hr_policy_queries": 60000,
            "sla_target": "Shortlist quality NDCG@10 >= 90%, EEOC 4/5ths adverse-impact violation rate == 0.0%, cross-band/country HR policy leak rate == 0.0%",
        },
        "tiers": [
            {
                "tier_id": "tier1_legacy_ats_exact_keyword_boolean_filter",
                "name": "Tier 1: Legacy ATS Exact Keyword Boolean Filter + Manual Recruiter Screening",
                "shortlist_hire_quality_ndcg_at_10_pct": 58.0,
                "eeoc_adverse_impact_violation_rate_pct": 6.5,
                "cross_country_hr_policy_leak_rate_pct": 4.2,
                "monthly_recruiting_and_hr_ops_cost_usd": 152000,
                "action_decision": "Send to a human",
            },
            {
                "tier_id": "tier2_unmasked_raw_resume_holistic_llm_scoring",
                "name": "Tier 2: Unmasked Raw Resume Fed Directly to Holistic 1-100 LLM Scorer",
                "shortlist_hire_quality_ndcg_at_10_pct": 74.5,
                "eeoc_adverse_impact_violation_rate_pct": 24.8,
                "cross_country_hr_policy_leak_rate_pct": 12.6,
                "monthly_recruiting_and_hr_ops_cost_usd": 215000,
                "action_decision": "Send to a human",
            },
            {
                "tier_id": "tier3_pii_redacted_resume_single_prompt_llm",
                "name": "Tier 3: PII-Redacted Resume + Single Holistic Prompt Scoring + Basic HR FAQ RAG",
                "shortlist_hire_quality_ndcg_at_10_pct": 81.2,
                "eeoc_adverse_impact_violation_rate_pct": 7.4,
                "cross_country_hr_policy_leak_rate_pct": 3.1,
                "monthly_recruiting_and_hr_ops_cost_usd": 68000,
                "action_decision": "Confirm",
            },
            {
                "tier_id": "tier4_pii_blind_esco_ontology_binary_rubric_eeoc_gate_rbac_copilot",
                "name": "Tier 4: PII/Proxy-Blind Parser + ESCO/O*NET Skill Ontology + Binary Competency Rubric + EEOC 4/5ths Audit Gate + Workday RBAC Policy Copilot",
                "shortlist_hire_quality_ndcg_at_10_pct": 93.6,
                "eeoc_adverse_impact_violation_rate_pct": 0.0,
                "cross_country_hr_policy_leak_rate_pct": 0.0,
                "monthly_recruiting_and_hr_ops_cost_usd": 23000,
                "action_decision": "Act automatically",
            },
        ],
        "perturbation_check": {
            "scenario": "Counterfactual Demographic Name/Gender/Graduation-Year Swap on Identical Senior Cloud Architect Resumes (NYC LL144 / EU AI Act Annex III Audit)",
            "tier2_unmasked_score_variance_and_bias_violation_pct": 24.8,
            "tier4_blind_binary_rubric_bias_violation_pct": 0.0,
            "tier1_keyword_ats_ndcg_pct": 58.0,
            "tier4_ontology_rubric_ndcg_pct": 93.6,
            "ndcg_gain_pts": 35.6,
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
