from experiments.enterprise_omnichannel_customer_service_oms_policy_gate_sizing.run import build_summary


def test_enterprise_omnichannel_customer_service_oms_policy_gate_sizing() -> None:
    s = build_summary()
    assert s["cost_reduction_pct_vs_human"] == 99.4
    assert s["cost_reduction_pct_vs_unconstrained_70b"] == 91.0
    assert s["gpu_reduction_pct_vs_unconstrained_70b"] == 81.3
    assert s["schemes"]["routed_8b_70b_plus_deterministic_oms_opa_gate"]["unauthorized_refund_leakage_pct"] == 0.0
