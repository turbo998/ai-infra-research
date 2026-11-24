from experiments.enterprise_multi_agent_mcp_top8_schema_router_durable_sizing.run import build_summary


def test_enterprise_multi_agent_mcp_top8_schema_router_durable_sizing() -> None:
    s = build_summary()
    opt = s["schemes"]["top8_mcp_router_structured_handoff_role_sticky_durable"]
    assert opt["input_tokens_per_step"] == 5200
    assert opt["workflow_10step_completion_pct"] == 97.8
    assert opt["prefix_kv_hit_rate_pct"] == 92.4
    assert opt["required_h200_gpus"] == 12
    assert s["latency_speedup_vs_baseline"] == 7.50
