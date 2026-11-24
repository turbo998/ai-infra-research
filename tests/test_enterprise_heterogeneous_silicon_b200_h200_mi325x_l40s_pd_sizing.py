from experiments.enterprise_heterogeneous_silicon_b200_h200_mi325x_l40s_pd_sizing.run import build_summary


def test_enterprise_heterogeneous_silicon_b200_h200_mi325x_l40s_pd_sizing() -> None:
    s = build_summary()
    opt = s["schemes"]["heterogeneous_pd_h100_prefill_h200_decode_l40s_tier1"]
    l40s = s["schemes"]["mismatched_l40s_tp4_colocated"]
    assert opt["p99_tpot_ms"] == 11.8
    assert opt["sla_goodput_pct"] == 99.4
    assert opt["cost_per_1m_goodput_tokens_usd"] == 0.082
    assert l40s["p99_tpot_ms"] == 56.3
    assert s["cost_savings_pct_vs_homogeneous_h200"] == 38.8
