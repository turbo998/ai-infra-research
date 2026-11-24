from experiments.enterprise_graphrag_vs_hybrid_vector_long_context_rag_sizing.run import evaluate_graphrag_router


def test_evaluate_graphrag_router():
    r = evaluate_graphrag_router(50, 1_000_000)
    assert r["lazy_graphrag_index_usd"] == 2700
    assert r["index_savings_pct"] == 94.0
    assert r["routed_monthly_query_usd"] == 7100
    assert r["blended_accuracy_pct"] == 93.8
