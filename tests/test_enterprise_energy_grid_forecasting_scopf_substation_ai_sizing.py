from experiments.enterprise_energy_grid_forecasting_scopf_substation_ai_sizing.run import evaluate_energy_grid_ai


def test_evaluate_energy_grid_ai():
    r = evaluate_energy_grid_ai(5000, 500_000)
    assert r["ai_probabilistic_mape_pct"] == 5.8
    assert r["escalated_images_per_day"] == 7500
    assert r["total_gpus"] == 6
    assert r["self_hosted_monthly_usd"] == 13300
    assert r["n_minus_1_physics_violations_pct"] == 0.0
    assert r["annual_imbalance_savings_million_usd"] == 9.4
