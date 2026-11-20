# Enterprise Cloud AI Landing Zone (`Hub-Spoke Private Link, Split-Horizon DNS & Egress Firewall`) (`R505`)

Deterministic quantitative evaluation model comparing four enterprise Cloud AI Landing Zone network security architectures across **30 application spoke VNets and 8 shared PaaS AI services**:
- **Tier 1**: Public Internet Endpoints (`publicNetworkAccess: Enabled`) with IP Whitelists Only
- **Tier 2**: Siloed Per-Spoke Private Endpoints (`30 Spokes x 8 Services = 240 Duplicate Private Endpoints`) + Disconnected DNS
- **Tier 3**: Forced Hairpin Routing of All High-Bandwidth GPU/Model-Weight Traffic Through Central L7 Deep-Packet-Inspection Firewall
- **Tier 4**: Hub-and-Spoke Shared AI Services VNet + Central `Private DNS Resolver` + Direct Spoke-to-AI `Private Link` + Outbound `FQDN` Egress Lock

## Verification

```bash
python experiments/enterprise_ai_landing_zone_hub_spoke_private_link_egress_firewall_selection/run.py --check
python -m unittest tests/test_enterprise_ai_landing_zone_hub_spoke_private_link_egress_firewall_selection.py
```
