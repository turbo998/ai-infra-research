# Enterprise RAG Vector Database & Index Selection Experiment (`Milvus` vs. `Qdrant` vs. `pgvectorscale` vs. `Azure AI Search`, `HNSW` vs. `DiskANN`)

Reproducible capacity and performance model comparing 4 vector indexing architectures on a `100M`-vector (`1536d`) enterprise RAG corpus across cluster DRAM footprint, NVMe SSD storage, `P99` query latency, `Recall@10`, and monthly infrastructure TCO.

## Run
```bash
python experiments/enterprise_rag_vector_db_milvus_qdrant_pgvector_ai_search_hnsw_diskann_selection/run.py
python experiments/enterprise_rag_vector_db_milvus_qdrant_pgvector_ai_search_hnsw_diskann_selection/run.py --check
```
