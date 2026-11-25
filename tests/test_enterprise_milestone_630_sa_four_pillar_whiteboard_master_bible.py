from experiments.enterprise_milestone_630_sa_four_pillar_whiteboard_master_bible.run import (
    evaluate_milestone_630_four_pillar_master_bible,
)


def test_milestone_630_four_pillar_master_bible():
    r = evaluate_milestone_630_four_pillar_master_bible()
    assert r["deliverable_id"] == "#630"
    assert r["four_pillar_monthly_usd"] == 50800
    assert r["tco_reduction_pct"] == 75.2
    assert r["annual_savings_usd"] == 1848000
    assert r["four_pillar_tpot_ms"] == 12.6
    assert r["four_pillar_goodput_pct"] == 99.5
