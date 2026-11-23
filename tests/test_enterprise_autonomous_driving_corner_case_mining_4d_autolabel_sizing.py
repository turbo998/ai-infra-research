from experiments.enterprise_autonomous_driving_corner_case_mining_4d_autolabel_sizing.run import build_summary


def test_enterprise_autonomous_driving_corner_case_mining_4d_autolabel_sizing() -> None:
    s = build_summary()
    assert s["upload_bandwidth_reduction_pct"] == 98.8
    assert s["labeling_cost_reduction_pct_vs_manual"] == 97.9
    assert s["turnaround_speedup_vs_manual"] == 77.54
    assert s["schemes"]["edge_shadow_trigger_plus_siglip_plus_4d_teacher_autolabel"]["corner_case_recall_pct"] == 98.2
