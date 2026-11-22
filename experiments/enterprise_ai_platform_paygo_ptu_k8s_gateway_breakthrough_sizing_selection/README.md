# Enterprise AI Platform End-to-End Architecture (`PAYGO` vs. `PTU` vs. `Self-Hosted K8s/AKS`), AI Gateway (`APIM / Envoy GIE / LiteLLM / Kong / Higress`) & 2026 AI Infra Breakthrough Radar Model (`#449`)

Reproducible analytical model comparing 100% Serverless PAYGO at scale, premature small-cluster self-hosting (with SRE staffing overhead), Managed PTU + APIM/Kong, and Two-Tier Gateway (`APIM North-South + Envoy GIE EPP East-West`) + Hybrid `PTU / SGLang / vLLM V1`.

- **Provenance**: `synthetic + anonymized_reported (independently_reproduced=false)`
- **Run**: `python experiments/enterprise_ai_platform_paygo_ptu_k8s_gateway_breakthrough_sizing_selection/run.py --check`
