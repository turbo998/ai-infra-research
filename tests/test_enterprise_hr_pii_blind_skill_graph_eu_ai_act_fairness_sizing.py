from experiments.enterprise_hr_pii_blind_skill_graph_eu_ai_act_fairness_sizing.run import build_summary


def test_enterprise_hr_pii_blind_skill_graph_eu_ai_act_fairness_sizing() -> None:
    s = build_summary()
    assert s["latency_speedup_vs_raw_70b"] == 18.89
    assert s["gpu_reduction_pct_vs_raw_70b"] == 87.5
    assert s["recall_gain_pts_vs_boolean_ats"] == 39.2
    assert s["schemes"]["pii_blind_esco_skill_graph_plus_auditable_rubric"]["demographic_bias_variance_pct"] == 0.3
