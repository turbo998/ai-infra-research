# Experiment: Enterprise Multi-Tenant LLM Inference Serving Stack Selection & Fully-Loaded TCO Crossover (`#404`)

Reproducible analytical model comparing `vLLM V1`, `SGLang (RadixAttention)`, `TensorRT-LLM`, and `Azure AI Foundry Managed Endpoints` across agentic throughput, Day-0 model onboarding agility, SRE engineering FTE overhead, and fully-loaded `$/1M tokens`.

- **Provenance**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)
- **Run**: `python experiments/enterprise_inference_serving_vllm_sglang_trtllm_azure_foundry_tco_selection/run.py`
- **Verify**: `python experiments/enterprise_inference_serving_vllm_sglang_trtllm_azure_foundry_tco_selection/run.py --check`
