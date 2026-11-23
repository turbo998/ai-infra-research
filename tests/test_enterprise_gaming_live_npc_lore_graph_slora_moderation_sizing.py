from experiments.enterprise_gaming_live_npc_lore_graph_slora_moderation_sizing.run import build_summary


def test_enterprise_gaming_live_npc_lore_graph_slora_moderation_sizing() -> None:
    s = build_summary()
    assert s["latency_speedup_vs_cloud_70b"] == 8.21
    assert s["gpu_reduction_pct_vs_cloud_70b"] == 91.4
    assert s["schemes"]["deberta_shield_plus_lore_graph_plus_slora_8b_plus_async_70b"]["p95_npc_response_ms"] == 95.0
    assert s["schemes"]["deberta_shield_plus_lore_graph_plus_slora_8b_plus_async_70b"]["lore_canon_consistency_pct"] == 99.2
