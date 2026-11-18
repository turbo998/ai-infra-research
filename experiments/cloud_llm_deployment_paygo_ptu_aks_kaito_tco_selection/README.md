# Enterprise Cloud Managed vs. Self-Hosted LLM Deployment Mode & TCO Selection (`cloud_llm_deployment_paygo_ptu_aks_kaito_tco_selection`)

Deterministic architectural evaluator for **SA Guide**: comparing **Serverless API (MaaS PAYGO)**, **Full-Peak Static Provisioned Throughput (Azure Foundry PTU / Bedrock Provisioned)**, **Self-Hosted Kubernetes GPU Cluster (AKS + KAITO / KEDA + vLLM)**, and **Hybrid Base-Capacity + Burst PAYGO Spillover via AI Gateway**.

## Usage

```bash
python experiments/cloud_llm_deployment_paygo_ptu_aks_kaito_tco_selection/run.py
python experiments/cloud_llm_deployment_paygo_ptu_aks_kaito_tco_selection/run.py --check
```
