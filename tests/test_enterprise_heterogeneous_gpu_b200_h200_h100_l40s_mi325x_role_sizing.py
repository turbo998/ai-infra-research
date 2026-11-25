from experiments.enterprise_heterogeneous_gpu_b200_h200_h100_l40s_mi325x_role_sizing.run import (
    evaluate_heterogeneous_gpu_fleet_role_sizing,
)


def test_heterogeneous_gpu_fleet_role_sizing():
    r = evaluate_heterogeneous_gpu_fleet_role_sizing()
    assert r["deliverable_id"] == "#629"
    assert r["heterogeneous_monthly_usd"] == 50800
    assert r["tco_savings_pct"] == 55.9
    assert r["three_year_savings_usd"] == 2318400
    assert r["heterogeneous_pd_tpot_ms"] == 12.6
    assert r["mi325x_kv_advantage_x"] == 3.86
