from experiments.enterprise_multi_lora_serving_slora_vs_icl_dedicated_sizing.run import evaluate_multi_lora


def test_evaluate_multi_lora():
    r = evaluate_multi_lora(100, 60, 32)
    assert r["adapter_size_mb"] == 268.4
    assert r["multi_lora_h200_gpus"] == 4
    assert r["multi_lora_monthly_usd"] == 13600
    assert r["savings_vs_dedicated_pct"] == 97.0
    assert r["savings_vs_icl_pct"] == 60.0
