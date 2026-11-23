from experiments.enterprise_milestone_540_multimodal_document_intelligence_vlm_routing_bible.run import build_summary


def test_enterprise_milestone_540_multimodal_document_intelligence_vlm_routing_bible() -> None:
    s = build_summary()
    assert s["latency_speedup_vs_monolithic_72b"] == 8.42
    assert s["gpu_reduction_pct_vs_monolithic_72b"] == 90.0
    assert s["cost_reduction_pct_vs_monolithic_72b"] == 96.2
    assert s["schemes"]["layout_router_plus_7b_vlm_plus_cross_sum_verifier"]["table_and_field_accuracy_pct"] == 98.6
