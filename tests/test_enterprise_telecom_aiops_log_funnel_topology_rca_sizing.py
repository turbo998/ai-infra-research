from experiments.enterprise_telecom_aiops_log_funnel_topology_rca_sizing.run import evaluate_telecom_aiops


def test_evaluate_telecom_aiops():
    r = evaluate_telecom_aiops(50, 494)
    assert r["stage1_drain3_bursts_per_sec"] == 420
    assert r["stage2_root_incidents_qps"] == 0.85
    assert r["rca_h200_gpus"] == 4
    assert r["funnel_monthly_usd"] == 15200
    assert r["savings_pct"] == 97.2
