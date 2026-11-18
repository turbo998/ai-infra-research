from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_payload() -> dict[str, Any]:
    return {
        "benchmark_id": "enterprise_erp_procure_to_pay_ap_three_way_match_agent_selection_v1",
        "title": "Enterprise ERP Procure-to-Pay (P2P) Accounts Payable 3-Way Match, UoM Normalization & Anti-BEC Bank-Master Gate Agent Evaluator",
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload_baseline": {
            "monthly_vendor_invoices": 280000,
            "active_global_suppliers": 14000,
            "monthly_po_and_goods_receipt_lines": 850000,
            "sla_target": "Touchless 3-Way Match STP >= 85%, duplicate payment / BEC bank-hijack rate == 0.0%, cost per invoice <= $1.80",
        },
        "tiers": [
            {
                "tier_id": "tier1_legacy_template_ocr_and_manual_ap_clerk_matching",
                "name": "Tier 1: Legacy Zonal OCR Templates + Manual AP Clerk 3-Way Matching in SAP MIRO",
                "touchless_three_way_match_stp_rate_pct": 26.5,
                "overpayment_duplicate_or_bec_fraud_rate_pct": 0.35,
                "cost_per_processed_invoice_usd": 11.80,
                "monthly_ap_processing_and_leakage_cost_usd": 330000,
                "action_decision": "Send to a human",
            },
            {
                "tier_id": "tier2_prompt_only_vlm_direct_sap_miro_posting",
                "name": "Tier 2: Prompt-Only VLM/LLM Reading Invoices & Directly Posting SAP Payment Vouchers",
                "touchless_three_way_match_stp_rate_pct": 78.0,
                "overpayment_duplicate_or_bec_fraud_rate_pct": 6.40,
                "cost_per_processed_invoice_usd": 16.50,
                "monthly_ap_processing_and_leakage_cost_usd": 462000,
                "action_decision": "Send to a human",
            },
            {
                "tier_id": "tier3_vlm_extraction_plus_vector_line_item_rag",
                "name": "Tier 3: VLM Invoice Extraction + Vector Similarity Line-Item Matching",
                "touchless_three_way_match_stp_rate_pct": 61.0,
                "overpayment_duplicate_or_bec_fraud_rate_pct": 1.85,
                "cost_per_processed_invoice_usd": 4.90,
                "monthly_ap_processing_and_leakage_cost_usd": 137000,
                "action_decision": "Confirm",
            },
            {
                "tier_id": "tier4_vlm_uom_normalizer_plus_deterministic_3way_tolerance_engine_plus_bec_bank_gate",
                "name": "Tier 4: Layout VLM + Semantic SKU/UoM Resolver + Deterministic 3-Way Match Tolerance Engine + Immutable SAP Vendor Bank-Master BEC Gate",
                "touchless_three_way_match_stp_rate_pct": 89.4,
                "overpayment_duplicate_or_bec_fraud_rate_pct": 0.0,
                "cost_per_processed_invoice_usd": 1.45,
                "monthly_ap_processing_and_leakage_cost_usd": 40600,
                "action_decision": "Act automatically",
            },
        ],
        "perturbation_check": {
            "scenario": "BEC Spoofed Invoice Injecting New 'Updated IBAN Account' Note + Pack-Size UoM Mismatch (PO: 1,200 EA @ $2.50 vs Invoice: 50 Cartons of 24 @ $60.00)",
            "tier2_bec_or_uom_error_rate_pct": 19.2,
            "tier4_bec_or_uom_error_rate_pct": 0.0,
            "tier1_stp_rate_pct": 26.5,
            "tier4_stp_rate_pct": 89.4,
            "stp_gain_pts": 62.9,
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
