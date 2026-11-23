# Enterprise `Serverless PAYGO` vs. `PTU` vs. `Self-Hosted K8s/AKS` TCO Inflection & AI Gateway Selection (`#512`)

Reproducible analytical model evaluating **three-stage LLM capacity procurement inflection points (`<22% Util PAYGO`, `22%–55% Util Base PTU + PAYGO Spillover`, `>=55% Util Self-Hosted AKS/K8s + vLLM V1 / SGLang`)** and comparing **five enterprise AI Gateways (`Azure APIM`, `LiteLLM`, `Kong AI Gateway`, `Envoy AI Gateway`, `Alibaba Higress`)**.

## Run

```bash
python experiments/enterprise_paygo_ptu_selfhosted_k8s_tco_ai_gateway_selection/run.py
```

- **Provenance**: `synthetic + anonymized_reported (independently_reproduced=false)`
- **Bilingual Guides**:
  - [Chinese FAQ](../../docs/components/enterprise-paygo-ptu-selfhosted-k8s-tco-ai-gateway-selection-faq.md)
  - [English FAQ](../../docs/components/enterprise-paygo-ptu-selfhosted-k8s-tco-ai-gateway-selection-faq.en.md)
- **Interactive H5 Lab**: [docs/h5/paygo-ptu-selfhosted-k8s-tco-ai-gateway-lab.html](../../docs/h5/paygo-ptu-selfhosted-k8s-tco-ai-gateway-lab.html)
