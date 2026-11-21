# Experiment: Enterprise Production RAG & Billion-Scale Vector Search Engine Selection (`#407`)

Reproducible analytical model comparing `Azure AI Search`, `Milvus 2.5 (DiskANN/CAGRA)`, `Qdrant`, and `PostgreSQL pgvectorscale` across `100M x 1536d` RAM/NVMe sizing, Hybrid `BM25 + Vector + Reranker` Recall@5, Pre-Filter ACL enforcement, and monthly TCO.

- **Provenance**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)
- **Run**: `python experiments/enterprise_rag_search_azure_ai_search_milvus_qdrant_pgvector_tco_selection/run.py`
- **Verify**: `python experiments/enterprise_rag_search_azure_ai_search_milvus_qdrant_pgvector_tco_selection/run.py --check`
