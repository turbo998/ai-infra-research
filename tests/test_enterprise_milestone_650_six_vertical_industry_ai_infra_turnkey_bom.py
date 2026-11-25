from experiments.enterprise_milestone_650_six_vertical_industry_ai_infra_turnkey_bom.run import (
    evaluate_milestone_650_vertical_blueprints,
)


def test_enterprise_milestone_650_six_vertical_industry_ai_infra_turnkey_bom() -> None:
    res = evaluate_milestone_650_vertical_blueprints()
    assert res["Deliverable_ID"] == "#650"
    assert res["Total_Bilingual_Playbooks"] == 652
    assert res["Total_Interactive_H5_Labs"] == 263
    assert len(res["Verticals"]) == 6
    assert res["Portfolio_Summary"]["combined_saved_3yr_m_usd"] == 55.59
    assert res["Portfolio_Summary"]["dual_slo_goodput_pct"] == 99.5
