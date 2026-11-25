from experiments.enterprise_milestone_640_sa_25_formulas_12_redlines_master_pocketbook.run import (
    evaluate_milestone_640_master_pocketbook,
)


def test_milestone_640_master_pocketbook():
    r = evaluate_milestone_640_master_pocketbook()
    assert r["deliverable_id"] == "#640"
    assert r["rounds_completed"] == 18
    assert r["golden_formulas_count"] == 25
    assert r["veto_redlines_count"] == 12
    assert r["unit_cost_reduction_x"] == 10.0
    assert r["unit_cost_savings_pct"] == 90.0
    assert r["compliant_goodput_pct"] == 99.5
