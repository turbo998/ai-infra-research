# Experiment #589: Enterprise Embedding & Reranker Serving (TEI/Infinity L40S & MRL INT8) Sizer

Reproduces the `4.28x` Top-50 Cross-Encoder reranking speedup (`14.5ms` vs. `62.0ms`) via FlashAttention `cu_seqlens` unpadded token packing on `L40S`, and the `8.0x` HNSW vector memory reduction (`80 GB` vs. `640 GB` for `100M` vectors) using Matryoshka `512d INT8` quantization (`docs/components/enterprise-embedding-reranker-serving-tei-infinity-l40s-sizing-faq.md`).
