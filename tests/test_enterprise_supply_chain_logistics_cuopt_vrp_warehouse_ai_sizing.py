from experiments.enterprise_supply_chain_logistics_cuopt_vrp_warehouse_ai_sizing.run import evaluate_supply_chain_logistics


def test_evaluate_supply_chain_logistics():
    r = evaluate_supply_chain_logistics(5000, 20_000)
    assert r["total_reopt_sec"] == 2.2
    assert r["l40s_gpus"] == 4
    assert r["h100_gpus"] == 4
    assert r["monthly_gpu_cost_usd"] == 13000
    assert r["annual_fuel_savings_million_usd"] == 11.2
    assert r["time_window_violation_pct"] == 0.0
