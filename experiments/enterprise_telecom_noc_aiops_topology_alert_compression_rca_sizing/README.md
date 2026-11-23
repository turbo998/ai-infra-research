# Enterprise Telecom & Cloud NOC AIOps (`100K Alerts/Sec`): `Direct Log-to-LLM` vs. `Vector Log RAG` vs. `CMDB Topology Graph Compression (99.4%) + Drain3 + 70B RCA Agent` (`#534`)

- **Chinese FAQ**: [`../../docs/components/enterprise-telecom-noc-aiops-topology-alert-compression-rca-sizing-faq.md`](../../docs/components/enterprise-telecom-noc-aiops-topology-alert-compression-rca-sizing-faq.md)
- **English FAQ**: [`../../docs/components/enterprise-telecom-noc-aiops-topology-alert-compression-rca-sizing-faq.en.md`](../../docs/components/enterprise-telecom-noc-aiops-topology-alert-compression-rca-sizing-faq.en.md)
- **Interactive H5 Lab**: [`../../docs/h5/telecom-noc-aiops-topology-alert-compression-rca-lab.html`](../../docs/h5/telecom-noc-aiops-topology-alert-compression-rca-lab.html)
- **Interactive Sizer**: [`../../docs/reports/enterprise-telecom-noc-aiops-topology-alert-compression-rca-sizing-sizer.html`](../../docs/reports/enterprise-telecom-noc-aiops-topology-alert-compression-rca-sizing-sizer.html)
- **7-Card Report**: [`../../docs/reports/enterprise-telecom-noc-aiops-topology-alert-compression-rca-sizing-cards.html`](../../docs/reports/enterprise-telecom-noc-aiops-topology-alert-compression-rca-sizing-cards.html)

## Reproduce
```bash
python experiments/enterprise_telecom_noc_aiops_topology_alert_compression_rca_sizing/run.py
python -m pytest tests/test_enterprise_telecom_noc_aiops_topology_alert_compression_rca_sizing.py -q
```
