from experiments.enterprise_pharma_biotech_molecular_clinical_trial_ectd_sizing.run import evaluate_pharma_clinical_ai


def test_evaluate_pharma_clinical_ai():
    r = evaluate_pharma_clinical_ai(250_000)
    assert r["llm_adjudicated_charts"] == 22000
    assert r["funnel_cost_usd"] == 410
    assert r["cost_savings_pct"] == 97.8
    assert r["funnel_hours"] == 1.4
    assert r["speedup_ratio"] == 27.1
