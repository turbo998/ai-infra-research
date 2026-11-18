# Enterprise Cybersecurity SOC Threat Hunting, MITRE ATT&CK Entity Graph, KQL/SPL Sandbox & SOAR Copilot Evaluator

Deterministic evaluator comparing four enterprise Security Operations Center (`SOC`) threat hunting and SOAR response architectures across 150,000 daily SIEM/EDR alerts, 65,000 endpoints, and 4,200 monthly security incidents:
- **Tier 1**: Legacy Static SIEM Sigma Rules + Manual L1 Analyst Triage (`68.0%` attribution accuracy, `0.5%` false isolation rate, `180.0 min` MTTC, `$320,000/mo`)
- **Tier 2**: Raw Log-Stuffed LLM with Unconstrained Autonomous SOAR Host/IP Isolation (`61.5%` attribution accuracy, `14.2%` critical-asset false isolation / log-injection rate, `12.0 min` MTTC, `$450,000/mo`)
- **Tier 3**: Standard Threat-Intel RAG + Unvalidated KQL/SPL Query Copilot (`82.4%` attribution accuracy, `2.8%` false isolation / query error rate, `42.0 min` MTTC, `$135,000/mo`)
- **Tier 4**: `OCSF` Entity Attack-Graph Correlation + AST-Validated Read-Only `KQL / SPL` Sandbox + Crown-Jewel Blast-Radius SOAR Gate (`96.2%` attribution accuracy, `0.0%` false isolation / log-injection rate, `6.5 min` MTTC, `$46,000/mo`)

## Usage
```bash
python experiments/enterprise_cybersecurity_soc_threat_hunting_soar_copilot_selection/run.py
python experiments/enterprise_cybersecurity_soc_threat_hunting_soar_copilot_selection/run.py --check
```
