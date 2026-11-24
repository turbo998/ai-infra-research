from experiments.enterprise_milestone_620_sa_fiftyeight_round_nineteen_industry_atlas_sizing.run import (
    evaluate_milestone_620_atlas,
)


def test_milestone_620_atlas():
    r = evaluate_milestone_620_atlas()
    assert r["deliverable_id"] == "#620"
    assert r["tonight_rounds_completed"] == 58
    assert r["total_repository_playbooks"] == 622
    assert r["vertical_industries_covered"] == 19
    assert r["reference_portfolio_monthly_usd"] == 844500
    assert r["portfolio_tco_savings_pct"] == 91.4
    assert r["post_verifier_violation_rate_pct"] == 0.0
