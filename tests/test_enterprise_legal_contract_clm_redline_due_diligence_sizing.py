from experiments.enterprise_legal_contract_clm_redline_due_diligence_sizing.run import evaluate_legal_clm


def test_evaluate_legal_clm():
    r = evaluate_legal_clm(25000, 18.0)
    assert r["token_reduction_pct"] == 76.4
    assert r["self_hosted_monthly_usd"] == 15500
    assert r["cost_per_contract_usd"] == 0.62
    assert r["savings_pct"] == 96.6
    assert r["risk_clause_recall_pct"] == 97.8
