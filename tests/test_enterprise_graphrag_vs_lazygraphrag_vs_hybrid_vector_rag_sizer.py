from experiments.enterprise_graphrag_vs_lazygraphrag_vs_hybrid_vector_rag_sizer.run import (
    evaluate_graphrag_vs_lazygraphrag,
)


def test_enterprise_graphrag_vs_lazygraphrag_vs_hybrid_vector_rag_sizer() -> None:
    res = evaluate_graphrag_vs_lazygraphrag(10_000_000)
    assert res["Deliverable_ID"] == "#663"
    idx = res["Indexing_Comparison"]
    assert idx["indexing_cost_saving_pct"] == 99.1
    qr = res["Query_Routing_82_18"]
    assert qr["lazygraphrag_multihop_acc_pct"] == 92.8
    assert qr["routed_weighted_latency_ms"] == 380
    assert qr["query_cost_saving_pct"] == 92.5
    assert qr["required_h200_gpus_100qps"] == 12
