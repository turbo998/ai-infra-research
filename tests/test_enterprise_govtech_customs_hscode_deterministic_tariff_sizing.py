from experiments.enterprise_govtech_customs_hscode_deterministic_tariff_sizing.run import build_summary


def test_enterprise_govtech_customs_hscode_deterministic_tariff_sizing() -> None:
    s = build_summary()
    assert s["gpu_reduction_pct_vs_unconstrained_70b"] == 83.3
    assert s["hscode_accuracy_gain_pts_vs_manual"] == 10.1
    assert s["schemes"]["hierarchical_hscode_plus_8b_70b_plus_deterministic_fta_engine"]["hscode_10digit_accuracy_pct"] == 98.9
    assert s["schemes"]["hierarchical_hscode_plus_8b_70b_plus_deterministic_fta_engine"]["tariff_calculation_error_pct"] == 0.0
