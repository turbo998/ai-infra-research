from experiments.enterprise_hr_talent_skills_graph_bias_audit_roleplay_sizing.run import evaluate_hr_talent_ai


def test_evaluate_hr_talent_ai():
    r = evaluate_hr_talent_ai(20_000, 500_000)
    assert r["total_l40s_gpus"] == 6
    assert r["roleplay_h100_gpus"] == 4
    assert r["self_hosted_monthly_usd"] == 14900
    assert r["savings_pct"] == 98.3
    assert r["disparate_impact_ratio_blind"] == 0.94
    assert r["counterfactual_name_swap_variance_pts"] == 0.0
