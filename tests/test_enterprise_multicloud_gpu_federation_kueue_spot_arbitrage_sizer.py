from experiments.enterprise_multicloud_gpu_federation_kueue_spot_arbitrage_sizer.run import (
    evaluate_multicloud_spot_arbitrage,
)


def test_enterprise_multicloud_gpu_federation_kueue_spot_arbitrage_sizer() -> None:
    res = evaluate_multicloud_spot_arbitrage(128)
    assert res["Deliverable_ID"] == "#662"
    tl = res["Preemption_Timeline_Sec"]
    assert tl["replacement_pod_ready_sec"] == 14.2
    assert tl["dropped_requests_pct"] == 0.0
    fin = res["FinOps_Comparison"]
    assert fin["rate_saving_pct"] == 54.7
    assert fin["three_az_provisioning_availability_pct"] == 99.92
