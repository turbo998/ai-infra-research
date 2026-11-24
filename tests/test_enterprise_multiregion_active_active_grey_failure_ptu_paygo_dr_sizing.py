from experiments.enterprise_multiregion_active_active_grey_failure_ptu_paygo_dr_sizing.run import build_summary


def test_enterprise_multiregion_active_active_grey_failure_ptu_paygo_dr_sizing() -> None:
    s = build_summary()
    opt = s["schemes"]["sticky_active_active_55_55_800ms_breaker_paygo_spillover"]
    assert opt["failover_rto_sec"] == 0.82
    assert opt["failover_p95_ttft_ms"] == 360.0
    assert opt["availability_sla_pct"] == 99.995
    assert opt["dr_capacity_overhead_pct"] == 10.0
    assert opt["annual_tco_usd"] == 1340000
