# `#646` Enterprise Billion-Scale Hybrid RAG (`HNSW` vs. `DiskANN` vs. `cuVS CAGRA`) & Two-Stage Reranker Sizer

Reproducible analytical verification for `#646`:
- **Chinese FAQ**: [`../../docs/components/enterprise-billion-scale-hybrid-rag-hnsw-diskann-cuvs-reranker-faq.md`](../../docs/components/enterprise-billion-scale-hybrid-rag-hnsw-diskann-cuvs-reranker-faq.md)
- **English FAQ**: [`../../docs/components/enterprise-billion-scale-hybrid-rag-hnsw-diskann-cuvs-reranker-faq.en.md`](../../docs/components/enterprise-billion-scale-hybrid-rag-hnsw-diskann-cuvs-reranker-faq.en.md)
- **Interactive H5 Lab**: [`../../docs/h5/billion-scale-hybrid-rag-hnsw-diskann-cuvs-reranker-lab.html`](../../docs/h5/billion-scale-hybrid-rag-hnsw-diskann-cuvs-reranker-lab.html)

```bash
python experiments/enterprise_billion_scale_hybrid_rag_hnsw_diskann_cuvs_reranker/run.py
pytest tests/test_enterprise_billion_scale_hybrid_rag_hnsw_diskann_cuvs_reranker.py -q
```
