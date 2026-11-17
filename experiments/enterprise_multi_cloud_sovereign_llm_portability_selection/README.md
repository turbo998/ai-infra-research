# Enterprise Sovereign Cloud, Multi-Cloud LLM Portability & Cross-Provider Failover Evaluator (Round 404)

Deterministic evaluator comparing four multi-cloud and sovereign AI portability architectures across 4 global regulatory jurisdictions (`US-East`, `EU-Frankfurt`, `CN-Shanghai`, `ME-Riyadh`) and 45,000,000 monthly requests:
- **Tier 1**: Single-Cloud Proprietary SDK Hardcoded in Microservices (`54.0%` sovereign compliance, `0.0%` failover consistency, `16.0 person-weeks` migration effort, `$64,000/mo`)
- **Tier 2**: Naive Multi-Cloud Proxy (Simple URL Swap Without Prompt/Schema Adaptation) (`88.0%` sovereign compliance, `68.5%` failover consistency, `6.0 person-weeks` effort, `$58,000/mo`)
- **Tier 3**: `100%` Dedicated `8xH100` Self-Hosted Clusters in Every Sovereign Region (`100.0%` sovereign compliance, `98.2%` failover consistency, `2.0 person-weeks` effort, `$96,000/mo`)
- **Tier 4**: Geo-Sovereign AI Gateway + Pre-Compiled Per-Model Prompt Adapters (`DSPy`) + Hybrid Open-Weights Sovereign Anchor (`100.0%` sovereign compliance, `97.6%` failover consistency, `0.5 person-weeks` effort, `$31,500/mo`)

## Usage
```bash
python experiments/enterprise_multi_cloud_sovereign_llm_portability_selection/run.py
python experiments/enterprise_multi_cloud_sovereign_llm_portability_selection/run.py --check
```
