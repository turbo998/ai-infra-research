from experiments.enterprise_reasoning_model_thinking_budget_router_kv_sizing.run import evaluate_reasoning_budget


def test_evaluate_reasoning_budget():
    r = evaluate_reasoning_budget(50)
    assert r["routed_out_tokens"] == 940
    assert r["token_reduction_pct"] == 75.6
    assert r["kv_reduction_pct"] == 74.2
    assert r["routed_h200_gpus"] == 8
    assert r["unconstrained_h200_gpus"] == 33 or r["unconstrained_h200_gpus"] == 32
