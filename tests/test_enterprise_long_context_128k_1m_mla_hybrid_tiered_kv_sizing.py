from experiments.enterprise_long_context_128k_1m_mla_hybrid_tiered_kv_sizing.run import build_summary


def test_enterprise_long_context_128k_1m_mla_hybrid_tiered_kv_sizing() -> None:
    s = build_summary()
    opt = s["schemes"]["mla_hybrid_2d_usp_3tier_rdma_kv_pool"]
    kv = s["kv_footprint_gb"]
    assert kv["deepseek_mla_fp8_128k"] == 4.61
    assert kv["hybrid_7to1_deltanet_128k"] == 2.68
    assert opt["reload_ttft_ms"] == 100.0
    assert opt["required_h200_gpus"] == 8
    assert s["reload_ttft_speedup_vs_recompute"] == 142.0
