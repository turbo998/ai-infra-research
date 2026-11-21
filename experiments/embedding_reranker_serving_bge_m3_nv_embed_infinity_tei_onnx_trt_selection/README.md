# Enterprise Embedding & Cross-Encoder Reranker Serving Selection Experiment (`HuggingFace TEI / Infinity` vs. `vLLM --task embed/score` vs. `Triton ONNX` vs. `Naive PyTorch`)

Reproducible latency, throughput, and GPU capacity model comparing 4 serving engines for online RAG Embedding + `Top-50` Cross-Encoder Reranking (`350 RAG QPS` = `17,500` rerank pairs/sec).

## Run
```bash
python experiments/embedding_reranker_serving_bge_m3_nv_embed_infinity_tei_onnx_trt_selection/run.py
python experiments/embedding_reranker_serving_bge_m3_nv_embed_infinity_tei_onnx_trt_selection/run.py --check
```
