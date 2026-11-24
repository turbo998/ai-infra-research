from experiments.enterprise_browser_computer_use_agent_dom_vs_vision_pool_sizing.run import evaluate_browser_agent_pool


def test_evaluate_browser_agent_pool():
    r = evaluate_browser_agent_pool(200, 15)
    assert r["hybrid_workflow_sec"] == 9.8
    assert r["pure_vision_workflow_sec"] == 36.0
    assert r["speedup_ratio"] == 3.67
    assert r["chromium_cpu_nodes"] == 13
    assert r["self_hosted_monthly_usd"] == 9100
    assert r["savings_pct"] == 85.6
