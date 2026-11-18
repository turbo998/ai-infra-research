# Enterprise Vector Database & Billion-Scale Hybrid Search Evaluator 

Deterministic analytical evaluator comparing four enterprise vector database and hybrid search architectures at a scale of `100,000,000` vectors (`1536-dim`) across `5,000` tenants (`300 QPS` peak):

1. **Tier 1 (`tier1_uncompressed_fp32_ram_hnsw`)**: Uncompressed FP32 In-Memory HNSW + 1-Index-Per-Tenant (`920 GB` RAM required, `94.5%` Recall@10, `68.0 ms` P95 latency, `$28,400/mo`).
2. **Tier 2 (`tier2_managed_cloud_search_paas`)**: Managed Cloud Enterprise Search PaaS (`Azure AI Search / Elastic Cloud`, `240 GB` RAM, `95.2%` Recall@10, `24.0 ms` P95 latency, `$13,500/mo`).
3. **Tier 3 (`tier3_quantized_milvus_qdrant_partition_key`)**: Self-Hosted Quantized Vector Cluster (`Milvus 2.5 / Qdrant` with Binary/SQ8 Quantization + Partition-Key Multi-Tenancy, `110 GB` RAM, `96.0%` Recall@10, `16.0 ms` P95 latency, `$5,800/mo`).
4. **Tier 4 (`tier4_tiered_ram_nvme_diskann_object_storage`)**: Hot/Warm/Cold Tiered Storage (`RAM + NVMe DiskANN/Mmap + Object Storage S3/Blob + Partition-Key Multi-Tenancy`, `48 GB` RAM [`19.2x` less RAM than Tier 1], `96.5%` Recall@10, `19.0 ms` P95 latency, `$2,400/mo` [`-91.5%` vs. Tier 1]).

## Usage

```bash
python experiments/enterprise_vector_database_billion_scale_hybrid_search_selection/run.py
python experiments/enterprise_vector_database_billion_scale_hybrid_search_selection/run.py --check
```
