from experiments.enterprise_paygo_ptu_k8s_tco_ai_gateway_breakthrough_radar.run import (
    evaluate_paygo_ptu_k8s_tco_gateway_radar,
)


def test_paygo_ptu_k8s_tco_gateway_radar():
    r = evaluate_paygo_ptu_k8s_tco_gateway_radar()
    assert r["deliverable_id"] == "#627"
    assert r["self_hosted_unit_usd_per_1m"] == 0.052
    assert r["cost_reduction_x"] == 9.23
    assert r["net_savings_pct"] == 89.2
    assert r["envoy_gie_prefix_hit_pct"] == 86.0
    assert r["envoy_gie_proxy_overhead_ms"] == 0.8
