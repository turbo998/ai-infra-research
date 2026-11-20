# Enterprise RAG Embedding & Cross-Encoder Reranker Serving Engine & Hardware Selection (`R493`)

Deterministic quantitative evaluation model comparing four RAG Embedding & Cross-Encoder Reranker serving architectures across **300 QPS online RAG traffic (50 candidates reranked per query) + 100M monthly ingested document chunks**:
- **Tier 1**: Unbatched PyTorch `FP16` Flask/FastAPI Wrapper on Expensive `H100` GPUs
- **Tier 2**: 100% Cloud Managed Embedding & Rerank API (`Cohere Rerank / OpenAI Embedding PAYGO`)
- **Tier 3**: CPU-Only `AVX-512 / AMX INT8 ONNX Runtime` for Both Embedding & 568M Cross-Encoder Reranking
- **Tier 4**: `L4 / L40S GPU` + `HuggingFace TEI` (FlashAttention + Token-Based Unpadded Batching) + `Matryoshka 512d` + Two-Stage `50->15` Rerank

## Verification

```bash
python experiments/enterprise_rag_embedding_reranker_serving_gpu_cpu_onnx_tei_selection/run.py --check
python -m unittest tests/test_enterprise_rag_embedding_reranker_serving_gpu_cpu_onnx_tei_selection.py
```
