from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_payload() -> dict[str, Any]:
    return {
        "benchmark_id": "enterprise_marketing_creative_brand_compliance_adtech_agent_selection_v1",
        "title": "Enterprise GenAI Marketing & Creative Supply Chain: Pixel-Locked Product Compositing, FTC/Ad-Law Claim Gate, C2PA & Contextual Bandit Agent Evaluator",
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload_baseline": {
            "monthly_localized_ad_creatives": 180000,
            "active_retail_cpg_skus": 25000,
            "monthly_personalized_impressions_millions": 80.0,
            "sla_target": "Brand & legal compliance pass rate >= 98.5%, FTC/Ad-Law or logo distortion rate == 0.0%, concept-to-launch <= 1.0 hour",
        },
        "tiers": [
            {
                "tier_id": "tier1_manual_agency_photoshoot_and_manual_legal_review",
                "name": "Tier 1: Manual Creative Agency Photoshoot/Retouching + Manual Legal Review",
                "brand_visual_and_copy_compliance_pass_rate_pct": 94.0,
                "ftc_ad_law_or_logo_hallucination_violation_rate_pct": 0.8,
                "campaign_concept_to_launch_hours": 120.0,
                "monthly_creative_agency_and_compliance_cost_usd": 310000,
                "action_decision": "Send to a human",
            },
            {
                "tier_id": "tier2_unconstrained_text_to_image_and_freeform_copy_llm",
                "name": "Tier 2: Unconstrained Text-to-Image Diffusion + Free-Form LLM Copy Direct Publishing",
                "brand_visual_and_copy_compliance_pass_rate_pct": 58.5,
                "ftc_ad_law_or_logo_hallucination_violation_rate_pct": 15.4,
                "campaign_concept_to_launch_hours": 0.8,
                "monthly_creative_agency_and_compliance_cost_usd": 395000,
                "action_decision": "Send to a human",
            },
            {
                "tier_id": "tier3_brand_lora_diffusion_plus_keyword_blocklist",
                "name": "Tier 3: Brand LoRA Fine-Tuned Diffusion + Static Keyword Blocklist",
                "brand_visual_and_copy_compliance_pass_rate_pct": 79.2,
                "ftc_ad_law_or_logo_hallucination_violation_rate_pct": 4.6,
                "campaign_concept_to_launch_hours": 6.5,
                "monthly_creative_agency_and_compliance_cost_usd": 135000,
                "action_decision": "Confirm",
            },
            {
                "tier_id": "tier4_locked_3d_foreground_iclight_plus_ftc_claim_matrix_c2pa_bandit_agent",
                "name": "Tier 4: Pixel-Locked SKU Foreground + IC-Light Relighting + Legal Claim-Substantiation Matrix + C2PA Signing + Fatigue-Capped Contextual Bandit Agent",
                "brand_visual_and_copy_compliance_pass_rate_pct": 99.2,
                "ftc_ad_law_or_logo_hallucination_violation_rate_pct": 0.0,
                "campaign_concept_to_launch_hours": 0.6,
                "monthly_creative_agency_and_compliance_cost_usd": 42000,
                "action_decision": "Act automatically",
            },
        ],
        "perturbation_check": {
            "scenario": "Skincare Serum Packaging Fine-Print Logo Preservation + Implied Clinical Efficacy Claim ('Erases Wrinkles in 24 Hours') across 15 Regional Jurisdictions",
            "tier2_violation_or_logo_warp_pct": 32.0,
            "tier4_pixel_lock_and_claim_gate_violation_pct": 0.0,
            "tier1_launch_hours": 120.0,
            "tier4_launch_hours": 0.6,
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
