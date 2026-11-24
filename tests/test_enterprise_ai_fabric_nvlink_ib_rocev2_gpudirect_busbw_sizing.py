from experiments.enterprise_ai_fabric_nvlink_ib_rocev2_gpudirect_busbw_sizing.run import build_summary


def test_enterprise_ai_fabric_nvlink_ib_rocev2_gpudirect_busbw_sizing() -> None:
    s = build_summary()
    roce = s["schemes"]["tuned_rail_rocev2_gpudirect_ecn_dlb"]
    untuned = s["schemes"]["untuned_cloud_ethernet_bounce_static_ecmp"]
    assert roce["measured_busbw_gbs"] == 47.6
    assert roce["line_rate_efficiency_pct"] == 95.2
    assert untuned["measured_busbw_gbs"] == 11.2
    assert s["busbw_speedup_vs_untuned"] == 4.25
    assert s["capex_savings_pct_vs_infiniband"] == 34.5
