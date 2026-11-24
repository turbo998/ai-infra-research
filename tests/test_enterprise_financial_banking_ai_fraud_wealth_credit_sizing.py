from experiments.enterprise_financial_banking_ai_fraud_wealth_credit_sizing.run import evaluate_banking_ai


def test_evaluate_banking_ai():
    r = evaluate_banking_ai(2500, 3000)
    assert r["inline_p99_ms"] == 29
    assert r["grey_zone_qps"] == 150
    assert r["held_sar_qps"] == 30
    assert r["total_gpus"] == 14
    assert r["unified_monthly_usd"] == 35600
    assert r["savings_pct"] == 94.5
