from experiments.enterprise_milestone_625_nvlink_rdma_ib_rocev2_lossless_busbw_sizing.run import (
    evaluate_milestone_625_fabric_busbw_sizing,
)


def test_milestone_625_fabric_busbw_sizing():
    r = evaluate_milestone_625_fabric_busbw_sizing()
    assert r["deliverable_id"] == "#625"
    assert r["tuned_pix_roce_busbw_gbs"] == 47.8
    assert r["line_rate_efficiency_pct"] == 95.6
    assert r["pix_vs_sys_speedup_x"] == 3.37
    assert r["rail_optics_reduction_pct"] == 56.2
    assert r["pfc_deadlock_rate_pct"] == 0.0
