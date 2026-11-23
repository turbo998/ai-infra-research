from experiments.enterprise_airline_irops_disruption_recovery_crew_milp_sizing.run import build_summary


def test_enterprise_airline_irops_disruption_recovery_crew_milp_sizing() -> None:
    s = build_summary()
    assert s["recovery_speedup_vs_manual"] == 21.67
    assert s["disruption_cost_reduction_pct"] == 74.0
    assert s["gpu_reduction_pct_vs_pure_70b"] == 75.0
    assert s["schemes"]["notam_70b_plus_colgen_milp_plus_8b_rebooking"]["faa_part117_legality_violation_pct"] == 0.0
