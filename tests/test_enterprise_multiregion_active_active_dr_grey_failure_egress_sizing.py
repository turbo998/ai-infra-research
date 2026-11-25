from experiments.enterprise_multiregion_active_active_dr_grey_failure_egress_sizing.run import (
    evaluate_multiregion_dr_grey_failure_egress_sizing,
)


def test_multiregion_dr_grey_failure_egress_sizing():
    r = evaluate_multiregion_dr_grey_failure_egress_sizing()
    assert r["deliverable_id"] == "#638"
    assert r["ttft_breaker_rto_sec"] == 0.85
    assert r["rto_speedup_x"] == 70.6
    assert r["dr_tco_savings_pct"] == 42.5
    assert r["egress_savings_pct"] == 99.1
    assert r["mid_stream_drop_error_pct"] == 0.0
