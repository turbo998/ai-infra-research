from experiments.enterprise_govcloud_smart_city_hotline_emergency_cad_sizing.run import evaluate_govcloud_smart_city


def test_evaluate_govcloud_smart_city():
    r = evaluate_govcloud_smart_city(10_000, 80_000)
    assert r["total_l40s_gpus"] == 16
    assert r["h200_gpus"] == 8
    assert r["total_gpus"] == 24
    assert r["govcloud_monthly_usd"] == 42400
    assert r["savings_pct"] == 89.8
    assert r["first_time_bureau_routing_accuracy_pct"] == 97.4
