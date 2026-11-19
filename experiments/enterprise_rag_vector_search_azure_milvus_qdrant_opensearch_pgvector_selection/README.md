# Enterprise RAG Vector Database & Hybrid Search Selection: `pgvectorscale` vs. `Azure AI Search / OpenSearch` vs. `Milvus / Qdrant` (`R468`)

Deterministic evaluation suite comparing four enterprise RAG vector search and hybrid retrieval architectures across an `80M-chunk` (`1536-d`) benchmark workload (`1,200 QPS`, `4,500` enterprise RBAC security groups):

1. **Tier 1**: Unindexed / Naive Post-Filtered Vector DB (`38.5%` filtered RBAC recall, `185.0 ms` P99 latency, `44.0%` exact SKU/code hit rate, `$68,000/mo` unquantized RAM TCO)
2. **Tier 2**: Existing Relational DB Vector Extension (`PostgreSQL pgvector + pgvectorscale DiskANN + BM25`) (`91.5%` filtered recall, `42.0 ms` P99 latency, `$14,000/mo` TCO — **Best for `<10M–20M` chunks**)
3. **Tier 3**: Turnkey Cloud Enterprise Search PaaS (`Azure AI Search / AWS OpenSearch`) (`96.8%` filtered recall, `28.0 ms` P99 latency, `97.4%` exact SKU hit rate, `$31,000/mo` TCO — **Best for turnkey enterprise RBAC & M365/Cloud integration**)
4. **Tier 4**: Dedicated Cloud-Native Vector Engine (`Milvus / Zilliz` or `Qdrant` Tiered NVMe/SQ8 + Bitmap Pre-Filter) (`98.4%` filtered recall, `16.5 ms` P99 latency, `97.8%` exact SKU hit rate, `$22,000/mo` TCO — **Best for `100M–1B+` high-QPS multi-tenant scale**)

## Usage

```bash
python experiments/enterprise_rag_vector_search_azure_milvus_qdrant_opensearch_pgvector_selection/run.py
python experiments/enterprise_rag_vector_search_azure_milvus_qdrant_opensearch_pgvector_selection/run.py --check
python -m unittest tests/test_enterprise_rag_vector_search_azure_milvus_qdrant_opensearch_pgvector_selection.py
```
