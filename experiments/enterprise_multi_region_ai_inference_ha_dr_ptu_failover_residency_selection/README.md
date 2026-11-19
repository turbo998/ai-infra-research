# Enterprise Multi-Region AI Inference `HA/DR`, Cross-Region `PTU/PAYGO` Failover & Data Residency Selection (`R474`)

Deterministic evaluation suite comparing four enterprise multi-region AI inference high-availability (`HA`), disaster recovery (`DR`), and data-residency routing architectures across a `200 PTU` (`EU / US / APAC`) global enterprise deployment:

1. **Tier 1**: Single-Region PTU / GPU Cluster Without Cross-Region Failover (`99.15%` availability, `3,600s` RTO, `48.5%` failover drops, `$145,000/mo` TCO incl. downtime penalty)
2. **Tier 2**: Active-Passive 100% Duplicated Idle Standby PTU Fleet (`2x 200 PTUs`) (`99.92%` availability, `45.0s` RTO, `$240,000/mo` TCO)
3. **Tier 3**: Unconstrained Global Round-Robin Load Balancing (`US <-> EU <-> APAC`) (`99.94%` availability, `34.0%` GDPR data residency violation rate, `$172,000/mo` TCO)
4. **Tier 4**: Geo-Fenced Active-Active PTU (`50/50` Dual-Region Inside Data Zone) + `TTFT/429` Circuit Breaker + Regional `PAYGO` Overflow (`99.995%` availability, `0.8s` brownout RTO, `0.0%` GDPR residency violations, `0.0%` failover drops, `$118,000/mo` TCO)

## Usage

```bash
python experiments/enterprise_multi_region_ai_inference_ha_dr_ptu_failover_residency_selection/run.py
python experiments/enterprise_multi_region_ai_inference_ha_dr_ptu_failover_residency_selection/run.py --check
python -m unittest tests/test_enterprise_multi_region_ai_inference_ha_dr_ptu_failover_residency_selection.py
```
