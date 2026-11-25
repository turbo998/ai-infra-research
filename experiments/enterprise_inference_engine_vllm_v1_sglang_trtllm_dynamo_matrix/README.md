# `#656` Enterprise LLM Inference Engine Selection Matrix (`vLLM V1` vs. `SGLang` vs. `TensorRT-LLM` vs. `NVIDIA Dynamo`) Model

Reproducible analytical verification for `#656`:
- **Chinese FAQ**: [`../../docs/components/enterprise-inference-engine-vllm-v1-sglang-trtllm-dynamo-matrix-faq.md`](../../docs/components/enterprise-inference-engine-vllm-v1-sglang-trtllm-dynamo-matrix-faq.md)
- **English FAQ**: [`../../docs/components/enterprise-inference-engine-vllm-v1-sglang-trtllm-dynamo-matrix-faq.en.md`](../../docs/components/enterprise-inference-engine-vllm-v1-sglang-trtllm-dynamo-matrix-faq.en.md)
- **Interactive H5 Lab**: [`../../docs/h5/inference-engine-vllm-v1-sglang-trtllm-dynamo-matrix-lab.html`](../../docs/h5/inference-engine-vllm-v1-sglang-trtllm-dynamo-matrix-lab.html)

```bash
python experiments/enterprise_inference_engine_vllm_v1_sglang_trtllm_dynamo_matrix/run.py
pytest tests/test_enterprise_inference_engine_vllm_v1_sglang_trtllm_dynamo_matrix.py -q
```
