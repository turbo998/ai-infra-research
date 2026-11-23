from experiments.enterprise_cybersecurity_soc_mitre_attack_graph_soar_sizing.run import build_summary


def test_enterprise_cybersecurity_soc_mitre_attack_graph_soar_sizing() -> None:
    s = build_summary()
    assert s["mttc_speedup_vs_raw_llm"] == 44.05
    assert s["gpu_reduction_pct_vs_raw_llm"] == 93.8
    assert s["lotl_recall_gain_pts_vs_static_siem"] == 38.2
    assert s["schemes"]["sigma_ueba_plus_mitre_graph_plus_70b_soar"]["mttc_containment_minutes"] == 4.2
