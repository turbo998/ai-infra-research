# Enterprise Vector Index Zero-Downtime Embedding Migration & Orthogonal Procrustes Adapter (`R459`)

Deterministic evaluation suite comparing four enterprise vector database (`Milvus / Qdrant / Weaviate / Azure AI Search / pgvector`) embedding-model upgrade (`v1 -> v2`) architectures across a `250M-chunk` production corpus (`180,000` daily live CDC updates, `1,500 QPS` search load):

1. **Tier 1**: Stop-the-World Index Drop + On-Demand GPU Re-Embedding (`2,280 min` [38h] downtime, `100.0%` outage recall drop, `12.4%` stale CDC miss, `$146,000` migration cost)
2. **Tier 2**: Naive In-Place Mixed-Model Vector Upsert (`0.0 min` downtime, `71.0%` Recall@10 collapse from incompatible vector spaces, `$185,000` cost)
3. **Tier 3**: Full Dual-RAM HNSW Shadow Index + Synchronous Dual-Encoder Write (`0.0 min` downtime, `1.2%` recall drop, `100.0%` peak RAM overhead, `$92,000` migration cost)
4. **Tier 4**: Orthogonal Procrustes Query Bridge (`W* = U V^T`) + SHA-256 Dedup Spot-GPU Backfill + Kafka CDC Watermark + Atomic Alias Swap (`0.0 min` downtime, `0.4%` transition recall drop, `0.0%` stale CDC miss, `18.0%` RAM overhead, `$29,000` migration TCO)

## Usage

```bash
python experiments/enterprise_vector_index_zero_downtime_embedding_migration_procrustes_selection/run.py
python experiments/enterprise_vector_index_zero_downtime_embedding_migration_procrustes_selection/run.py --check
python -m unittest tests/test_enterprise_vector_index_zero_downtime_embedding_migration_procrustes_selection.py
```
