# Enterprise Billion-Scale Vector Search & Hybrid RAG Infrastructure (`In-Memory HNSW` vs. `SQ8/BQ` vs. `DiskANN NVMe` vs. `GPU cuVS CAGRA`) Memory & TCO Sizing (`#467`)

Reproducible analytical model comparing uncompressed FP32 HNSW in DRAM, SQ8/Binary Quantized HNSW, DiskANN/Vamana SSD-resident graph indexing, and NVIDIA cuVS CAGRA GPU-accelerated index build + tiered serving for `1 Billion` `1536-d` vectors.

- **Provenance**: `synthetic + anonymized_reported (independently_reproduced=false)`
- **Run**: `python experiments/enterprise_rag_vector_search_hnsw_diskann_cuvs_memory_tco_sizing_selection/run.py --check`
