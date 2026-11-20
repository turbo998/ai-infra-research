# Enterprise Multi-Cloud AI Active-Active Disaster Recovery & Cross-Provider Failover Selection (`R492`)

Deterministic quantitative evaluation model comparing four multi-cloud AI high-availability (HA) and disaster-recovery (DR) architectures across **80B monthly enterprise tokens**:
- **Tier 1**: Single-Cloud Single-Region Endpoint with Client Exponential Backoff (`30s–60s` retry loop)
- **Tier 2**: Dual-Cloud 100% Duplicated Active-Passive Reserved Capacity (`2x Full PTU / Provisioned Throughput`)
- **Tier 3**: Naive Round-Robin Multi-Cloud Proxy Without Prompt/Tool Canonicalization or TTFT Probe
- **Tier 4**: Weighted Active-Active Multi-Cloud AI Gateway (`70% Primary Cloud PTU + 20% Secondary Cloud + 10% Self-Hosted K8s vLLM`) + `900ms TTFT` Hedged Failover + Canonical Prompt/Schema Adapter

## Verification

```bash
python experiments/enterprise_multi_cloud_ai_disaster_recovery_active_active_gateway_failover_selection/run.py --check
python -m unittest tests/test_enterprise_multi_cloud_ai_disaster_recovery_active_active_gateway_failover_selection.py
```
