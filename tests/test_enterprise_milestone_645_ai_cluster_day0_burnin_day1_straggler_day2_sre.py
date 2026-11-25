from experiments.enterprise_milestone_645_ai_cluster_day0_burnin_day1_straggler_day2_sre.run import (
    evaluate_cluster_sre_lifecycle,
)


def test_enterprise_milestone_645_ai_cluster_day0_burnin_day1_straggler_day2_sre() -> None:
    res = evaluate_cluster_sre_lifecycle(1024)
    assert res["Deliverable_ID"] == "#645"
    assert res["Day0_BurnIn_5Gates"]["doa_units_intercepted"] == 39
    assert res["Day1_Canaries"]["straggler_compute_ratio_threshold"] == 1.08
    d2 = res["Day2_Auto_Healing"]
    assert d2["auto_mttr_min"] == 3.2
    assert d2["auto_availability_pct"] == 99.4
    assert d2["equivalent_gpus_recovered"] == 133
