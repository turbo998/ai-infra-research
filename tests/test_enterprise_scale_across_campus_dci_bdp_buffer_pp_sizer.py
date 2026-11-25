from experiments.enterprise_scale_across_campus_dci_bdp_buffer_pp_sizer.run import (
    compute_pfc_headroom_kb,
    evaluate_scale_across_dci,
)


def test_enterprise_scale_across_campus_dci_bdp_buffer_pp_sizer() -> None:
    res = evaluate_scale_across_dci()
    assert res["Deliverable_ID"] == "#659"
    assert compute_pfc_headroom_kb(5.0, 400.0) == 2500
    c5 = res["Campus_5km_Analysis"]
    assert c5["required_pfc_headroom_per_port_kb"] == 2500
    assert c5["deep_buffer_pp_dp_efficiency_pct"] == 94.2
    assert c5["cross_building_pd_disagg_efficiency_pct"] == 96.8
