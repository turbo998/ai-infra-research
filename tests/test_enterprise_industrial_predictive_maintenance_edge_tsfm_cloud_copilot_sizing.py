from experiments.enterprise_industrial_predictive_maintenance_edge_tsfm_cloud_copilot_sizing.run import build_summary


def test_enterprise_industrial_predictive_maintenance_edge_tsfm_cloud_copilot_sizing() -> None:
    s = build_summary()
    assert s["wan_bandwidth_reduction_pct"] == 99.7
    assert s["mttr_speedup_vs_static_plc"] == 7.11
    assert s["central_gpu_reduction_pct_vs_cloud_raw"] == 87.5
    assert s["schemes"]["edge_fft_timesfm_plus_onprem_70b_schematic_copilot"]["early_fault_recall_pct"] == 96.4
