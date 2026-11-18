from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_payload() -> dict[str, Any]:
    return {
        "benchmark_id": "enterprise_insurance_claims_triage_underwriting_fraud_agent_selection_v1",
        "title": "Enterprise Insurance Claims Triage, Multimodal Invoice Forensics, GNN Collusion Fraud Detection & Deterministic Actuarial Adjudication Agent Evaluator",
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload_baseline": {
            "monthly_pc_and_health_claims": 420000,
            "monthly_invoice_and_damage_images": 1800000,
            "active_policy_wordings_and_riders": 1200,
            "sla_target": "Straight-Through Processing (STP) >= 80%, fraud & forged-invoice recall >= 95%, payout & clause accuracy >= 99%, 100% human adjuster sign-off on denials",
        },
        "tiers": [
            {
                "tier_id": "tier1_manual_claims_adjuster_and_legacy_rule_engine",
                "name": "Tier 1: Manual Claims Adjuster Review + Legacy Hardcoded Rule Engine",
                "straight_through_processing_stp_rate_pct": 22.0,
                "fraud_and_forged_invoice_detection_recall_pct": 58.5,
                "payout_calculation_and_clause_accuracy_pct": 93.5,
                "monthly_claims_ops_and_fraud_leakage_cost_usd": 390000,
                "action_decision": "Send to a human",
            },
            {
                "tier_id": "tier2_end_to_end_vlm_prompt_adjudication_and_mental_payout",
                "name": "Tier 2: End-to-End Prompted VLM/LLM Directly Deciding Coverage & Payout Amount",
                "straight_through_processing_stp_rate_pct": 74.0,
                "fraud_and_forged_invoice_detection_recall_pct": 46.0,
                "payout_calculation_and_clause_accuracy_pct": 68.4,
                "monthly_claims_ops_and_fraud_leakage_cost_usd": 480000,
                "action_decision": "Send to a human",
            },
            {
                "tier_id": "tier3_vlm_invoice_ocr_plus_standard_policy_vector_rag",
                "name": "Tier 3: VLM Invoice/Damage OCR + Standard Policy Wording Vector RAG",
                "straight_through_processing_stp_rate_pct": 58.5,
                "fraud_and_forged_invoice_detection_recall_pct": 71.2,
                "payout_calculation_and_clause_accuracy_pct": 83.6,
                "monthly_claims_ops_and_fraud_leakage_cost_usd": 175000,
                "action_decision": "Confirm",
            },
            {
                "tier_id": "tier4_pixel_forensics_gnn_fraud_plus_policy_ast_actuarial_engine_siu_agent",
                "name": "Tier 4: Pixel/EXIF Forensics + GNN Collusion Graph + Policy Rider AST + Deterministic Actuarial Payout Calculator + SIU/Adjuster HITL Agent",
                "straight_through_processing_stp_rate_pct": 82.5,
                "fraud_and_forged_invoice_detection_recall_pct": 96.4,
                "payout_calculation_and_clause_accuracy_pct": 99.6,
                "monthly_claims_ops_and_fraud_leakage_cost_usd": 62000,
                "action_decision": "Act automatically",
            },
        ],
        "perturbation_check": {
            "scenario": "AIGC/Photoshopped Hospital Receipt Font-Clone Forgery + Multi-Rider Deductible/Co-Pay/Annual-Cap Waterfall Calculation",
            "tier2_vlm_fraud_recall_pct": 38.0,
            "tier4_forensics_gnn_fraud_recall_pct": 95.8,
            "fraud_recall_gain_pts": 57.8,
            "tier2_payout_accuracy_pct": 61.5,
            "tier4_actuarial_engine_payout_accuracy_pct": 99.6,
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
