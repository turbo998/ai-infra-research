from experiments.enterprise_telecom_noc_aiops_topology_alert_compression_rca_sizing.run import build_summary


def test_enterprise_telecom_noc_aiops_topology_alert_compression_rca_sizing() -> None:
    s = build_summary()
    assert s["mttr_speedup_vs_direct_llm"] == 17.5
    assert s["gpu_reduction_pct_vs_direct_llm"] == 93.8
    assert s["schemes"]["topology_causal_graph_plus_drain3_plus_70b_rca"]["alert_compression_pct"] == 99.4
    assert s["schemes"]["topology_causal_graph_plus_drain3_plus_70b_rca"]["top1_rca_accuracy_pct"] == 94.8
