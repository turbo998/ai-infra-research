# Enterprise Cloud LLM Deployment Selection: Serverless PAYGO vs. Provisioned PTU/GSU vs. Self-Hosted K8s (AKS/GKE) TCO Benchmark

Deterministic quantitative evaluation model for [`docs/components/cloud-deployment-paygo-vs-provisioned-ptu-gsu-vs-self-hosted-k8s-tco-selection-faq.md`](../../docs/components/cloud-deployment-paygo-vs-provisioned-ptu-gsu-vs-self-hosted-k8s-tco-selection-faq.md) and [`.en.md`](../../docs/components/cloud-deployment-paygo-vs-provisioned-ptu-gsu-vs-self-hosted-k8s-tco-selection-faq.en.md).

## Provenance
- **Label**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)

## Run & Verify
```bash
python experiments/cloud_deployment_paygo_vs_provisioned_ptu_gsu_vs_self_hosted_k8s_tco_selection/run.py
python experiments/cloud_deployment_paygo_vs_provisioned_ptu_gsu_vs_self_hosted_k8s_tco_selection/run.py --check
python -m unittest tests/test_cloud_deployment_paygo_vs_provisioned_ptu_gsu_vs_self_hosted_k8s_tco_selection.py
```
