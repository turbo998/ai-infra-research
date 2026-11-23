from experiments.enterprise_ai_security_guardrails_spotlighting_obo_tool_policy_sizing.run import build_summary


def test_enterprise_ai_security_guardrails_spotlighting_obo_tool_policy_sizing() -> None:
    s = build_summary()
    assert s["latency_speedup_vs_70b_judge"] == 33.04
    assert s["gpu_reduction_pct_vs_70b_judge"] == 91.7
    assert s["schemes"]["deberta_spotlighting_obo_opa_zero_trust"]["unauthorized_tool_exec_pct"] == 0.0
    assert s["schemes"]["deberta_spotlighting_obo_opa_zero_trust"]["indirect_injection_bypass_pct"] == 0.12
