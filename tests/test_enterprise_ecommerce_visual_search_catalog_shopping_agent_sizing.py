from experiments.enterprise_ecommerce_visual_search_catalog_shopping_agent_sizing.run import evaluate_ecommerce_ai


def test_evaluate_ecommerce_ai():
    r = evaluate_ecommerce_ai(1500, 200, 2.0)
    assert r["search_l40s_gpus"] == 8
    assert r["chat_8b_l40s_gpus"] == 4
    assert r["total_l40s_gpus"] == 12
    assert r["chat_72b_h200_gpus"] == 6
    assert r["unified_monthly_usd"] == 32300
    assert r["savings_pct"] == 92.5
