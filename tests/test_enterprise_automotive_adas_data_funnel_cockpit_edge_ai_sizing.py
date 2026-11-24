from experiments.enterprise_automotive_adas_data_funnel_cockpit_edge_ai_sizing.run import (
    evaluate_automotive_adas_cockpit_sizing,
)


def test_automotive_adas_cockpit_sizing():
    r = evaluate_automotive_adas_cockpit_sizing()
    assert r["deliverable_id"] == "#617"
    assert r["uploaded_pb_per_month"] == 2.25
    assert r["volume_reduction_pct"] == 99.5
    assert r["cockpit_edge_ttft_ms"] == 18
    assert r["optimized_monthly_usd"] == 324000
    assert r["tco_savings_pct"] == 88.6
