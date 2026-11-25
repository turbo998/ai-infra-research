from experiments.enterprise_pd_disaggregation_dynamic_pd_ratio_kv_backpressure_sizing.run import (
    evaluate_dynamic_pd_disaggregation_sizing,
)


def test_dynamic_pd_disaggregation_sizing():
    r = evaluate_dynamic_pd_disaggregation_sizing()
    assert r["deliverable_id"] == "#623"
    assert r["dynamic_pd_p99_tpot_ms"] == 12.8
    assert r["tpot_speedup_x"] == 5.34
    assert r["dynamic_pd_gpus"] == 16
    assert r["gpu_savings_vs_colocated_pct"] == 50.0
    assert r["gpu_savings_vs_static_pct"] == 33.3
    assert r["dynamic_pd_oom_drop_pct"] == 0.0
    assert r["goodput_pct"] == 99.5
