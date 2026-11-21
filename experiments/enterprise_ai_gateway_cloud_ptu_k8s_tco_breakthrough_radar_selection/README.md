# Experiment: Enterprise AI Platform Selection & 2026 Breakthrough Radar — PAYGO vs. PTU vs. Self-Hosted K8s & AI Gateways (`#400`)

Reproducible analytical model evaluating the three TCO crossover points (`Serverless PAYGO` vs. `Provisioned PTU` vs. `Self-Hosted K8s + vLLM/SGLang`), five AI Gateway architectures (`Azure APIM`, `Envoy AI Gateway / K8s GIE`, `LiteLLM`, `Kong AI`, `Higress`), and 2026 AI Infra capability breakthroughs.

- **Provenance**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)
- **Run**: `python experiments/enterprise_ai_gateway_cloud_ptu_k8s_tco_breakthrough_radar_selection/run.py`
- **Verify**: `python experiments/enterprise_ai_gateway_cloud_ptu_k8s_tco_breakthrough_radar_selection/run.py --check`
