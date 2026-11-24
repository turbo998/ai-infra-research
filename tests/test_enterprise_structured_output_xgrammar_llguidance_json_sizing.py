from experiments.enterprise_structured_output_xgrammar_llguidance_json_sizing.run import evaluate_structured_output


def test_evaluate_structured_output():
    r = evaluate_structured_output(100, 320)
    assert r["active_decode_steps"] == 235
    assert r["p99_speedup_ratio"] == 4.57
    assert r["xgrammar_h200_gpus"] == 8
    assert r["prompt_retry_h200_gpus"] == 12
    assert r["gpu_savings_pct"] == 33.3
