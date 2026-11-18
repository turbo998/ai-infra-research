# Enterprise Multi-Region High Availability (HA/DR) & Cross-Cloud LLM Traffic Routing (`enterprise_multi_region_ha_dr_cross_cloud_routing_selection`)

Deterministic architectural evaluator for **SA Guide**: comparing **Single-Region Unprotected Deployment**, **Active-Passive 1:1 Idle Standby**, **Active-Active Multi-Region Weighted Pool + Global PAYGO Spillover**, and **Active-Active Multi-Cloud/Hybrid + P95 Tail-Latency Request Hedging**.

## Usage

```bash
python experiments/enterprise_multi_region_ha_dr_cross_cloud_routing_selection/run.py
python experiments/enterprise_multi_region_ha_dr_cross_cloud_routing_selection/run.py --check
```
