from experiments.enterprise_financial_10k_earnings_point_in_time_python_sandbox_sizing.run import build_summary


def test_enterprise_financial_10k_earnings_point_in_time_python_sandbox_sizing() -> None:
    s = build_summary()
    assert s["latency_speedup_vs_128k"] == 8.57
    assert s["gpu_reduction_pct_vs_128k"] == 87.5
    assert s["schemes"]["xbrl_table_pit_filter_plus_python_sandbox"]["arithmetic_error_rate_pct"] == 0.0
    assert s["schemes"]["xbrl_table_pit_filter_plus_python_sandbox"]["temporal_point_in_time_accuracy_pct"] == 99.4
