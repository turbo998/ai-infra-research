# Enterprise Private & Hybrid-Cloud AI Landing Zone Architecture Selection (`R480`)

Deterministic evaluation suite comparing four enterprise private, on-premises, and hybrid-cloud AI Landing Zone architectures across `25,000` employees (`20%` `L4` Crown-Jewel/Restricted data, `80%` `L1–L3` Internal/Confidential workloads):

1. **Tier 1**: Public Internet SaaS LLM Endpoints Without Private Link or Data Classification (`0.0%` L4 sovereignty compliance, `$195,000/mo` risk + ops TCO)
2. **Tier 2**: 100% Air-Gapped On-Premises Physical GPU Data Center for All `L1–L4` Workloads (`100.0%` L4 compliance, `68.0%` frontier capability coverage, `62.5%` hardware CapEx/idle waste, `$238,000/mo` amortized TCO)
3. **Tier 3**: Cloud Private Link Only While Querying On-Premises Core Databases Synchronously Over WAN (`1,890ms` P95 cross-WAN RAG latency, `$146,000/mo` TCO)
4. **Tier 4**: Data-Classified Hybrid AI Landing Zone (`20%` `L4` on Local `8x H200` On-Premises/Arc Cluster + `80%` `L1–L3` over `ExpressRoute / Direct Connect Private Link` + `CMK` Zero-Retention Cloud AI) (`100.0%` L4 compliance, `99.2%` frontier capability coverage, `290ms` P95 latency, `8.5%` waste, `$68,000/mo` TCO)

## Usage

```bash
python experiments/enterprise_hybrid_cloud_private_ai_landing_zone_selection/run.py
python experiments/enterprise_hybrid_cloud_private_ai_landing_zone_selection/run.py --check
python -m unittest tests/test_enterprise_hybrid_cloud_private_ai_landing_zone_selection.py
```
