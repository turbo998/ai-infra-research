from experiments.enterprise_reasoning_think_budget_soft_close_kv_stripping.run import (
    evaluate_reasoning_think_budget_kv_stripping,
)


def test_reasoning_think_budget_kv_stripping():
    r = evaluate_reasoning_think_budget_kv_stripping()
    assert r["deliverable_id"] == "#633"
    assert r["routed_avg_out_tokens"] == 940
    assert r["output_token_reduction_pct"] == 75.6
    assert r["soft_close_truncation_pct"] == 0.0
    assert r["routed_h200_gpus"] == 8
    assert r["gpu_savings_pct"] == 75.0
