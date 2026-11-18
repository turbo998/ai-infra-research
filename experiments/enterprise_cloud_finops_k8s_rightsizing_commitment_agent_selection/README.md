# Enterprise Multi-Cloud FinOps (FOCUS 1.0), JVM-Aware Kubernetes Rightsizing & LP Commitment Portfolio Evaluator

Deterministic evaluator comparing four enterprise multi-cloud FinOps, Kubernetes rightsizing, and RI/Savings Plan commitment optimization architectures across `$2.5M/month` cloud spend, 120 EKS/AKS/GKE clusters, and 65,000 production pods:
- **Tier 1**: Monthly Spreadsheet Bill Review + Manual Ad-Hoc RI/Savings-Plan Buying (`11.5%` cloud savings, `1.2%` OOM/stranded rate, `168.0 h` anomaly lag, `$640,000/mo` remaining waste & ops cost)
- **Tier 2**: Naive Average-Utilization LLM Directly Cutting K8s Limits & Buying 3-Year RIs (`19.0%` savings, `14.8%` `OOMKilled` / stranded RI rate, `2.5 h` lag, `$580,000/mo`)
- **Tier 3**: Cloud Cost Explorer RAG + Vanilla K8s VPA Recommendations (`23.4%` savings, `4.2%` OOM/stranded rate, `24.0 h` lag, `$310,000/mo`)
- **Tier 4**: `FinOps FOCUS 1.0` Semantic Layer + Runtime-Aware (`JVM Heap / Go GC P99.9`) Safe K8s Rightsizing + Linear-Programming (`LP`) Commitment Ladder + Terraform GitOps Agent (`38.6%` cloud savings, `0.0%` `OOMKilled` / stranded commitment rate, `0.5 h` anomaly remediation, `$95,000/mo`)

## Usage
```bash
python experiments/enterprise_cloud_finops_k8s_rightsizing_commitment_agent_selection/run.py
python experiments/enterprise_cloud_finops_k8s_rightsizing_commitment_agent_selection/run.py --check
```
