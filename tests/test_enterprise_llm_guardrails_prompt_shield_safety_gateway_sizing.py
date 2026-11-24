from experiments.enterprise_llm_guardrails_prompt_shield_safety_gateway_sizing.run import evaluate_guardrail_gateway


def test_evaluate_guardrail_gateway():
    r = evaluate_guardrail_gateway(100, 240)
    assert r["parallel_penalty_ms"] == 6
    assert r["serial_penalty_ms"] == 340
    assert r["l40s_gpus"] == 2
    assert r["self_hosted_monthly_usd"] == 1900
    assert r["indirect_injection_block_rate_spotlighting_pct"] == 98.6
