from experiments.enterprise_agriculture_satellite_drone_crop_vlm_yield_ai_sizing.run import (
    evaluate_agriculture_satellite_drone_sizing,
)


def test_agriculture_satellite_drone_sizing():
    r = evaluate_agriculture_satellite_drone_sizing()
    assert r["deliverable_id"] == "#618"
    assert r["optimized_cloud_vlm_monthly_usd"] == 24800
    assert r["tco_savings_pct"] == 92.0
    assert r["herbicide_reduction_pct"] == 68.4
    assert r["epa_phi_dosage_violation_pct"] == 0.0
    assert r["yield_forecast_mape_pct"] == 2.8
