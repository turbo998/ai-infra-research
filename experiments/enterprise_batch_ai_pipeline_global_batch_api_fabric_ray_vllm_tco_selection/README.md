# Enterprise Batch AI Pipeline Selection (`50%-Off Global Batch API` vs. `Microsoft Fabric AI Functions` vs. `Ray Data + vLLM Offline Batch`)

Reproducible analytical model comparing four enterprise batch AI pipeline architectures processing 10 million records per night (`12B tokens/night`) across `Prefix KV cache hit rate`, `per-GPU roofline throughput (tok/s)`, `completion time`, and `monthly TCO`.

- **Provenance**: `synthetic + anonymized_reported (independently_reproduced=false)`
- **Run**: `python experiments/enterprise_batch_ai_pipeline_global_batch_api_fabric_ray_vllm_tco_selection/run.py`
