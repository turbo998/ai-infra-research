from experiments.enterprise_coding_assistant_fim_swe_agent_sandbox_sizing.run import evaluate_coding_platform


def test_evaluate_coding_platform():
    r = evaluate_coding_platform(1000)
    assert r["debounced_fim_qps"] == 28.0
    assert r["fim_h100_gpus"] == 4
    assert r["swe_h200_gpus"] == 8
    assert r["kata_sandbox_nodes"] == 4
    assert r["self_hosted_monthly_usd"] == 39600
