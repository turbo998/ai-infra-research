from experiments.enterprise_billion_scale_hybrid_rag_hnsw_diskann_cuvs_reranker.run import (
    evaluate_billion_scale_rag,
)


def test_enterprise_billion_scale_hybrid_rag_hnsw_diskann_cuvs_reranker() -> None:
    res = evaluate_billion_scale_rag()
    assert res["Deliverable_ID"] == "#646"
    idx = res["Index_Memory_Comparison"]
    assert idx["diskann_pq64_ram_gb"] == 192
    assert idx["ram_saving_vs_fp32_pct"] == 96.1
    rr = res["Two_Stage_Hybrid_Rerank"]
    assert rr["total_retrieval_p95_ms"] == 18.4
    assert rr["ndcg_at_5_pct"] == 93.8
    assert rr["llm_h200_gpus_with_rerank"] == 16
