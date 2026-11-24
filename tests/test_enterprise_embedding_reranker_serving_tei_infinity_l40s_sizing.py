from experiments.enterprise_embedding_reranker_serving_tei_infinity_l40s_sizing.run import evaluate_embedding_reranker


def test_evaluate_embedding_reranker():
    r = evaluate_embedding_reranker(200, 100)
    assert r["padding_waste_pct"] == 56.2
    assert r["speedup_ratio"] == 4.28
    assert r["l40s_gpus"] == 2
    assert r["mrl_512d_int8_hnsw_gb"] == 80
    assert r["ram_compression_ratio"] == 8.0
    assert r["total_self_hosted_monthly_usd"] == 2380
