# LLM Inference Serving Engine Selection: vLLM V1 vs. SGLang vs. TensorRT-LLM vs. NVIDIA Dynamo vs. Ollama/llama.cpp

Deterministic quantitative evaluation model for [`docs/components/inference-serving-engine-vllm-sglang-trtllm-dynamo-llamacpp-selection-faq.md`](../../docs/components/inference-serving-engine-vllm-sglang-trtllm-dynamo-llamacpp-selection-faq.md) and [`.en.md`](../../docs/components/inference-serving-engine-vllm-sglang-trtllm-dynamo-llamacpp-selection-faq.en.md).

## Provenance
- **Label**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)

## Run & Verify
```bash
python experiments/inference_serving_engine_vllm_sglang_trtllm_dynamo_llamacpp_selection/run.py
python experiments/inference_serving_engine_vllm_sglang_trtllm_dynamo_llamacpp_selection/run.py --check
python -m unittest tests/test_inference_serving_engine_vllm_sglang_trtllm_dynamo_llamacpp_selection.py
```
