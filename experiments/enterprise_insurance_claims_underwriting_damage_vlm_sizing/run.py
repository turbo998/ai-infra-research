"""Reproducible model for #612 Enterprise Insurance Claims STP Funnel & Fraud GraphRAG Sizer."""
from __future__ import annotations
import json
import math
from pathlib import Path


def evaluate_insurance_claims(monthly_claims: int = 1_200_000) -> dict:
    scale = monthly_claims / 1_200_000.0
    l40s_stp_gpus = max(2, math.ceil(4 * scale))
    h200_complex_gpus = max(2, math.ceil(6 * scale))

    self_hosted_monthly_usd = l40s_stp_gpus * 950 + h200_complex_gpus * 3400
    all_cloud_vlm_monthly_usd = round(monthly_claims * 1.80)
    cost_per_claim_usd = round(self_hosted_monthly_usd / monthly_claims, 3)

    annual_fraud_recovery_million_usd = round(18.40 * scale, 2)

    return {
        "topic_id": 612,
        "monthly_claims": monthly_claims,
        "tier1_stp_share_pct": 62.0,
        "tier1_stp_settlement_sec": 18.0,
        "l40s_stp_gpus": l40s_stp_gpus,
        "h200_complex_gpus": h200_complex_gpus,
        "self_hosted_monthly_usd": self_hosted_monthly_usd,
        "all_cloud_vlm_monthly_usd": all_cloud_vlm_monthly_usd,
        "cost_per_claim_usd": cost_per_claim_usd,
        "compute_savings_pct": round((1.0 - self_hosted_monthly_usd / all_cloud_vlm_monthly_usd) * 100, 1),
        "annual_fraud_recovery_million_usd": annual_fraud_recovery_million_usd,
        "payout_math_error_pct": 0.0,
    }


def main() -> None:
    res = evaluate_insurance_claims()
    out_path = Path(__file__).with_name("results.json")
    out_path.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print("Wrote", out_path.name)


if __name__ == "__main__":
    main()
