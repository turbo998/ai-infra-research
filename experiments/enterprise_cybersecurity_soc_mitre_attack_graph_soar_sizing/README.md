# Enterprise Cybersecurity SOC Threat Hunting & Automated SOAR (`50B Events/Day`): `Raw Log-to-LLM` vs. `Static SIEM Rules` vs. `Sigma/UEBA Filter (99.8%) + MITRE ATT&CK Kill-Chain Graph + 70B SOAR Investigator` (`#557`)

- **Chinese FAQ**: [`../../docs/components/enterprise-cybersecurity-soc-mitre-attack-graph-soar-sizing-faq.md`](../../docs/components/enterprise-cybersecurity-soc-mitre-attack-graph-soar-sizing-faq.md)
- **English FAQ**: [`../../docs/components/enterprise-cybersecurity-soc-mitre-attack-graph-soar-sizing-faq.en.md`](../../docs/components/enterprise-cybersecurity-soc-mitre-attack-graph-soar-sizing-faq.en.md)
- **Interactive H5 Lab**: [`../../docs/h5/cybersecurity-soc-mitre-attack-graph-soar-lab.html`](../../docs/h5/cybersecurity-soc-mitre-attack-graph-soar-lab.html)
- **Interactive Sizer**: [`../../docs/reports/enterprise-cybersecurity-soc-mitre-attack-graph-soar-sizing-sizer.html`](../../docs/reports/enterprise-cybersecurity-soc-mitre-attack-graph-soar-sizing-sizer.html)
- **7-Card Report**: [`../../docs/reports/enterprise-cybersecurity-soc-mitre-attack-graph-soar-sizing-cards.html`](../../docs/reports/enterprise-cybersecurity-soc-mitre-attack-graph-soar-sizing-cards.html)

## Reproduce
```bash
python experiments/enterprise_cybersecurity_soc_mitre_attack_graph_soar_sizing/run.py
python -m pytest tests/test_enterprise_cybersecurity_soc_mitre_attack_graph_soar_sizing.py -q
```
