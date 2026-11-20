# Enterprise RAG Multi-Stage Retrieval & Reranking Cascade Architecture Selection Benchmark

Deterministic quantitative evaluation model for [`docs/components/enterprise-rag-reranker-cascade-hybrid-bm25-colbert-cross-encoder-selection-faq.md`](../../docs/components/enterprise-rag-reranker-cascade-hybrid-bm25-colbert-cross-encoder-selection-faq.md) and [`.en.md`](../../docs/components/enterprise-rag-reranker-cascade-hybrid-bm25-colbert-cross-encoder-selection-faq.en.md).

## Provenance
- **Label**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)

## Run & Verify
```bash
python experiments/enterprise_rag_reranker_cascade_hybrid_bm25_colbert_cross_encoder_selection/run.py
python experiments/enterprise_rag_reranker_cascade_hybrid_bm25_colbert_cross_encoder_selection/run.py --check
python -m unittest tests/test_enterprise_rag_reranker_cascade_hybrid_bm25_colbert_cross_encoder_selection.py
```
