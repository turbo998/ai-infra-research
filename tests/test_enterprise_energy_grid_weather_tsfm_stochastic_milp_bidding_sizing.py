from experiments.enterprise_energy_grid_weather_tsfm_stochastic_milp_bidding_sizing.run import build_summary


def test_enterprise_energy_grid_weather_tsfm_stochastic_milp_bidding_sizing() -> None:
    s = build_summary()
    assert s["imbalance_penalty_reduction_pct"] == 64.3
    assert s["weather_update_speedup_vs_nwp"] == 480.0
    assert s["bess_arbitrage_revenue_lift_pct"] == 23.8
    assert s["schemes"]["graphcast_timesfm_plus_70b_plus_stochastic_milp"]["grid_constraint_violation_pct"] == 0.0
