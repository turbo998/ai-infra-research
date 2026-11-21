# LLM Inference Memory & Graph Runtime: PagedAttention/FlashInfer, Piece-wise CUDA Graph & Persistent Cache Benchmark

Deterministic quantitative evaluation model for [`docs/components/llm-inference-pagedattention-flashinfer-cudagraph-memory-pool-selection-faq.md`](../../docs/components/llm-inference-pagedattention-flashinfer-cudagraph-memory-pool-selection-faq.md) and [`.en.md`](../../docs/components/llm-inference-pagedattention-flashinfer-cudagraph-memory-pool-selection-faq.en.md).

## Provenance
- **Label**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)

## Run & Verify
```bash
python experiments/llm_inference_pagedattention_flashinfer_cudagraph_memory_pool_selection/run.py
python experiments/llm_inference_pagedattention_flashinfer_cudagraph_memory_pool_selection/run.py --check
python -m unittest tests/test_llm_inference_pagedattention_flashinfer_cudagraph_memory_pool_selection.py
```
