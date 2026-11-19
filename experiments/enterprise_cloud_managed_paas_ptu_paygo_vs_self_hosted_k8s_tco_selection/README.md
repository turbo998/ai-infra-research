# Enterprise Cloud-Managed (`PAYGO` vs. `PTU`) vs. Self-Hosted Kubernetes (`AKS/EKS + KAITO/vLLM`) TCO Selection (`R489`)

Deterministic evaluation suite comparing four enterprise LLM deployment modes across `120 Billion` monthly tokens (`40%` frontier proprietary models, `60%` high-volume open-weight/distilled models, including fully loaded `$18,500/mo` per Platform SRE FTE payroll):

1. **Tier 1**: 100% Serverless `PAYGO` Token API at High Volume (`$2.65/1M tokens`, `2,380ms` P99 TTFT, `14.5%` peak HTTP 429 throttling, `0.2 FTE`, `$322,000/mo` full TCO)
2. **Tier 2**: 100% Peak-Sized Cloud Provisioned Throughput (`PTU` / Bedrock Provisioned) (`$1.95/1M tokens`, `480ms` P99 TTFT, `0.5 FTE`, `$244,000/mo` full TCO)
3. **Tier 3**: Raw DIY Self-Hosted `Kubernetes + vLLM` with 3 Full-Time GPU SREs (`$1.62/1M tokens`, `3.0 SRE FTEs = $55.5K/mo payroll`, `$194,000/mo` full TCO)
4. **Tier 4**: Volume-Inflection Hybrid (`P55 Base PTU + PAYGO Burst` for Frontier + `AKS KAITO + vLLM` for `>30B Tok/Mo` Open-Weight) (`$0.72/1M tokens`, `420ms` P99 TTFT, `0.0%` HTTP 429 throttling, `1.0 SRE FTE`, `$106,000/mo` full TCO)

## Usage

```bash
python experiments/enterprise_cloud_managed_paas_ptu_paygo_vs_self_hosted_k8s_tco_selection/run.py
python experiments/enterprise_cloud_managed_paas_ptu_paygo_vs_self_hosted_k8s_tco_selection/run.py --check
python -m unittest tests/test_enterprise_cloud_managed_paas_ptu_paygo_vs_self_hosted_k8s_tco_selection.py
```
