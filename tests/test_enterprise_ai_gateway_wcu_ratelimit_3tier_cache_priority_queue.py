from experiments.enterprise_ai_gateway_wcu_ratelimit_3tier_cache_priority_queue.run import (
    compute_wcu_per_sec,
    evaluate_gateway_governance,
)


def test_enterprise_ai_gateway_wcu_ratelimit_3tier_cache_priority_queue() -> None:
    res = evaluate_gateway_governance()
    assert res["Deliverable_ID"] == "#641"
    assert compute_wcu_per_sec(100.0, 4000, 500, 0.0, 0.185) == 533400.0
    opt = res["Optimized_WCU_3Tier_QoS"]
    assert opt["wcu_per_sec"] < 240000.0
    assert opt["required_h200_gpus"] == 18
    assert opt["gpu_saving_pct"] == 43.8
    assert opt["p0_p99_ttft_ms"] == 285
    assert opt["p0_goodput_pct"] == 99.6
