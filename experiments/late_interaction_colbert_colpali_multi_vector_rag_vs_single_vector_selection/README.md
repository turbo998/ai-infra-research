# Late-Interaction Multi-Vector RAG Selection Experiment (`ColBERTv2 / ColPali / ColQwen2` vs. `Single-Vector Bi-Encoder` & `2-Stage Token-Pooled Binary MaxSim`)

Reproducible index memory footprint, `ViDoRe` visual/table `Recall@5`, query `P99` latency, and TCO model comparing 4 retrieval architectures across a `5,000,000`-page enterprise document corpus.

## Run
```bash
python experiments/late_interaction_colbert_colpali_multi_vector_rag_vs_single_vector_selection/run.py
python experiments/late_interaction_colbert_colpali_multi_vector_rag_vs_single_vector_selection/run.py --check
```
