# Experiment: Enterprise AI Gateway Token Quota Governance & Multi-BU FinOps Chargeback (`#412`)

Reproducible analytical model evaluating `Legacy RPM Limiting`, `Static max_tokens Pre-Deduction`, and `Two-Phase 4-Tuple Weighted Token Buckets + Elastic Idle Quota Borrowing` across `Azure APIM`, `Kong AI Gateway`, `LiteLLM Enterprise`, and `Envoy AI Gateway`.

- **Provenance**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)
- **Run**: `python experiments/enterprise_ai_gateway_token_quota_chargeback_apim_kong_litellm_envoy_selection/run.py`
- **Verify**: `python experiments/enterprise_ai_gateway_token_quota_chargeback_apim_kong_litellm_envoy_selection/run.py --check`
