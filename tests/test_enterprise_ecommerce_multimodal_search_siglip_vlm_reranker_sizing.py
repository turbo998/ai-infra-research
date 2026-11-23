from experiments.enterprise_ecommerce_multimodal_search_siglip_vlm_reranker_sizing.run import build_summary


def test_enterprise_ecommerce_multimodal_search_siglip_vlm_reranker_sizing() -> None:
    s = build_summary()
    assert s["latency_speedup_vs_direct_72b"] == 16.09
    assert s["gpu_reduction_pct_vs_direct_72b"] == 92.5
    assert s["schemes"]["siglip2_ann_plus_7b_vlm_reranker_plus_gated_70b"]["visual_recall_at_10_pct"] == 93.8
    assert s["schemes"]["siglip2_ann_plus_7b_vlm_reranker_plus_gated_70b"]["required_h200_gpus"] == 24
