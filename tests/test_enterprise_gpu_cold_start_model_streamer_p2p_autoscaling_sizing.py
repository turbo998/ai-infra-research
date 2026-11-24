from experiments.enterprise_gpu_cold_start_model_streamer_p2p_autoscaling_sizing.run import evaluate_cold_start_autoscaling


def test_evaluate_cold_start_autoscaling():
    r = evaluate_cold_start_autoscaling(70.0, 32)
    assert r["legacy_total_sec"] == 510.0
    assert r["fast_total_sec"] == 17.5
    assert r["speedup_ratio"] == 29.1
    assert r["fast_fleet_gpus"] == 36
    assert r["legacy_fleet_gpus"] == 52
    assert r["monthly_savings_usd"] == 36800
