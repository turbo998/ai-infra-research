# Enterprise Telecom 5G RAN/Core NOC Alarm Storm RCA, Netconf/YANG & Digital-Twin Canary Agent Evaluator

Deterministic evaluator comparing four telecom Network Operations Center (`NOC`) root-cause analysis (`RCA`) and autonomous network (`TM Forum L4`) remediation architectures across 45,000 5G gNodeB base stations, 1,200 metro/core nodes, and 1.2M daily network alarms:
- **Tier 1**: Legacy Static NOC Alarm Correlation Rules + Manual Engineer CLI Triage (`64.5%` cross-layer RCA accuracy, `1.8%` config error rate, `95.0 min` MTTR, `$340,000/mo`)
- **Tier 2**: Raw Alarm Stream Stuffed into LLM with Direct Unverified CLI/Netconf Push (`56.0%` RCA accuracy, `13.6%` config outage / syntax error rate, `38.0 min` MTTR, `$510,000/mo`)
- **Tier 3**: Standard 3GPP / Vendor Manual Vector RAG + Read-Only Ticket Copilot (`78.5%` RCA accuracy, `4.2%` CLI dialect drift rate, `45.0 min` MTTR, `$155,000/mo`)
- **Tier 4**: Cross-Layer Topology Causal Graph (`L1 Optical -> L3 IP -> 5G RAN/Core`) + `Netconf/YANG` Compiler + Network Digital-Twin Simulation & 60s Canary Auto-Rollback Agent (`95.8%` RCA accuracy, `0.0%` config outage rate, `8.2 min` MTTR, `$49,000/mo`)

## Usage
```bash
python experiments/enterprise_telecom_5g_network_noc_rca_digital_twin_agent_selection/run.py
python experiments/enterprise_telecom_5g_network_noc_rca_digital_twin_agent_selection/run.py --check
```
