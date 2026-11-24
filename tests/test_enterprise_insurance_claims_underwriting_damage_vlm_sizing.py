from experiments.enterprise_insurance_claims_underwriting_damage_vlm_sizing.run import evaluate_insurance_claims


def test_evaluate_insurance_claims():
    r = evaluate_insurance_claims(1_200_000)
    assert r["l40s_stp_gpus"] == 4
    assert r["h200_complex_gpus"] == 6
    assert r["self_hosted_monthly_usd"] == 24200
    assert r["cost_per_claim_usd"] == 0.02
    assert r["compute_savings_pct"] == 98.9
    assert r["payout_math_error_pct"] == 0.0
