from experiments.enterprise_cold_chain_logistics_fefo_shelf_life_cuopt_milp_sizing.run import build_summary


def test_enterprise_cold_chain_logistics_fefo_shelf_life_cuopt_milp_sizing() -> None:
    s = build_summary()
    assert s["spoilage_reduction_pct_vs_fifo"] == 68.9
    assert s["gross_margin_lift_pct"] == 11.4
    assert s["gpu_reduction_pct_vs_freeform_llm"] == 75.0
    assert s["schemes"]["iot_kinetic_shelf_life_plus_fefo_cuopt_milp"]["route_constraint_violation_pct"] == 0.0
