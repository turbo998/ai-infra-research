from experiments.enterprise_multitenant_rag_acl_semantic_prefix_kv_gateway_sizing.run import build_summary


def test_enterprise_multitenant_rag_acl_semantic_prefix_kv_gateway_sizing() -> None:
    s = build_summary()
    opt = s["schemes"]["pre_filter_acl_entity_l1_envoy_gie_l2_prefix_kv"]
    assert opt["restricted_tenant_recall_at_10_pct"] == 96.4
    assert opt["semantic_cache_false_positive_pct"] == 0.02
    assert opt["engine_prefix_kv_hit_rate_pct"] == 91.5
    assert opt["p95_ttft_ms"] == 210.0
    assert opt["required_h200_gpus"] == 12
