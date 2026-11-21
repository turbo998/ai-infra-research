# Enterprise AI Gateway Deep Dive: Azure APIM vs. LiteLLM Proxy vs. Kong AI vs. Envoy AI Gateway / Higress Benchmark

Deterministic quantitative evaluation model for [`docs/components/enterprise-ai-gateway-deep-dive-apim-litellm-kong-envoy-higress-selection-faq.md`](../../docs/components/enterprise-ai-gateway-deep-dive-apim-litellm-kong-envoy-higress-selection-faq.md) and [`.en.md`](../../docs/components/enterprise-ai-gateway-deep-dive-apim-litellm-kong-envoy-higress-selection-faq.en.md).

## Provenance
- **Label**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)

## Run & Verify
```bash
python experiments/enterprise_ai_gateway_deep_dive_apim_litellm_kong_envoy_higress_selection/run.py
python experiments/enterprise_ai_gateway_deep_dive_apim_litellm_kong_envoy_higress_selection/run.py --check
python -m unittest tests/test_enterprise_ai_gateway_deep_dive_apim_litellm_kong_envoy_higress_selection.py
```
