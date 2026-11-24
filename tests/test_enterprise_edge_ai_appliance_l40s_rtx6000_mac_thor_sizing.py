from experiments.enterprise_edge_ai_appliance_l40s_rtx6000_mac_thor_sizing.run import evaluate_edge_appliance


def test_evaluate_edge_appliance():
    r = evaluate_edge_appliance(10, 50)
    assert r["compute_ratio"] == 20.7
    assert r["concurrent_ttft_l40s_ms"] == 846 or r["concurrent_ttft_l40s_ms"] == 920
    assert r["concurrent_ttft_mac_ms"] == 42000
    assert r["dual_l40s_max_recommended_users"] == 24
