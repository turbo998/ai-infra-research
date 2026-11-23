from experiments.enterprise_milestone_555_caio_roi_npv_stage_gated_finops_bible.run import build_summary


def test_enterprise_milestone_555_caio_roi_npv_stage_gated_finops_bible() -> None:
    s = build_summary()
    assert s["unit_cost_reduction_pct_vs_api_sprawl"] == 82.3
    assert s["three_year_npv_lift_millions_vs_capex_first"] == 13.8
    assert s["payback_months_stage_gated"] == 5.8
    assert s["schemes"]["stage_gated_paygo_to_ptu_cascade_to_pd_k8s"]["three_year_roi_pct"] == 284.0
