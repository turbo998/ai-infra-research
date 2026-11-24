from experiments.enterprise_ai_platform_paygo_ptu_k8s_gateway_radar_sizing.run import build_summary


def test_enterprise_ai_platform_paygo_ptu_k8s_gateway_radar_sizing() -> None:
    s = build_summary()
    opt = s["schemes"]["hybrid_crossover_2tier_gie_gateway_tidal_k8s"]
    tc = s["tco_crossover_thresholds"]
    assert tc["paygo_to_ptu_utilization_pct"] == 24.0
    assert tc["ptu_to_self_hosted_k8s_utilization_pct"] == 56.0
    assert opt["prefix_kv_hit_rate_pct"] == 91.8
    assert opt["cost_per_1m_tokens_usd"] == 0.058
    assert s["cost_reduction_factor_vs_pure_paygo"] == 8.59
