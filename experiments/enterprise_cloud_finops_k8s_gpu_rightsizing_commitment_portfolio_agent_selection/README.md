# Enterprise Cloud & AI FinOps Agent (`FOCUS 1.0`, `P99.9` VRAM/RAM Rightsizing & `RI/PTU` Portfolio) Benchmark

Deterministic quantitative evaluation model for [`docs/components/enterprise-cloud-finops-k8s-gpu-rightsizing-commitment-portfolio-agent-selection-faq.md`](../../docs/components/enterprise-cloud-finops-k8s-gpu-rightsizing-commitment-portfolio-agent-selection-faq.md) and [`.en.md`](../../docs/components/enterprise-cloud-finops-k8s-gpu-rightsizing-commitment-portfolio-agent-selection-faq.en.md).

## Provenance
- **Label**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)

## Run & Verify
```bash
python experiments/enterprise_cloud_finops_k8s_gpu_rightsizing_commitment_portfolio_agent_selection/run.py
python experiments/enterprise_cloud_finops_k8s_gpu_rightsizing_commitment_portfolio_agent_selection/run.py --check
python -m unittest tests/test_enterprise_cloud_finops_k8s_gpu_rightsizing_commitment_portfolio_agent_selection.py
```
