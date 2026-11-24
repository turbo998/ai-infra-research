from experiments.enterprise_milestone_610_sa_twentyfive_domains_compendium_sizing.run import evaluate_milestone_610


def test_evaluate_milestone_610():
    r = evaluate_milestone_610()
    assert r["topic_id"] == 610
    assert r["deliverables_synthesized"] == 48
    assert r["domains_covered"] == 25
    assert r["total_vertical_optimized_monthly_usd"] == 291200
    assert r["portfolio_savings_pct"] == 93.4
    assert r["post_verifier_domain_violation_pct"] == 0.0
