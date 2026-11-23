from experiments.enterprise_wealth_management_suitability_tax_loss_optimizer_sizing.run import build_summary


def test_enterprise_wealth_management_suitability_tax_loss_optimizer() -> None:
    s = build_summary()
    assert s["gpu_reduction_pct_vs_freeform_70b"] == 75.0
    assert s["after_tax_alpha_lift_bps"] == 142.0
    assert s["suitability_compliance_pct"] == 100.0
    assert s["schemes"]["intent_70b_plus_convex_tax_lot_optimizer_plus_reg_bi_gate"]["reg_bi_suitability_violation_pct"] == 0.0
