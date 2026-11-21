# Experiment: Enterprise Multi-Model Cascade Router & 3-Tier Caching Gateway Selection (`#406`)

Reproducible analytical model evaluating `78/22` Multi-Model Cascade Routing, Entity-Guarded Semantic Caching vs. Naive Cosine Caching (`8.2%` false-positive trap), and K8s `GIE EPP` Prefix KV Routing across `Azure APIM`, `Envoy AI Gateway`, `LiteLLM`, and `Higress`.

- **Provenance**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)
- **Run**: `python experiments/enterprise_multi_model_router_prompt_semantic_cache_gateway_tco_selection/run.py`
- **Verify**: `python experiments/enterprise_multi_model_router_prompt_semantic_cache_gateway_tco_selection/run.py --check`
