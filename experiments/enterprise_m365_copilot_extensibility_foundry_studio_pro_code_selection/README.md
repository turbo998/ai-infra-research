# Enterprise Copilot Extensibility & Build-vs-Buy Architecture Selection: `M365 Declarative Agents` vs. `Copilot Studio` vs. Pro-Code `Azure AI Foundry` (`R490`)

Deterministic evaluation suite comparing four enterprise AI portfolio architectures across `30` departmental and enterprise AI use cases (`50%` workplace knowledge & CRUD, `30%` departmental workflow automation, `20%` complex custom-model & multi-agent workflows):

1. **Tier 1**: Building 30 Custom Pro-Code Standalone Web Chat Portals from Scratch (`24.5%` employee daily adoption, `62.0%` SharePoint/Graph ACL fidelity, `16.0 weeks` time-to-prod, `$265,000/mo` build + run TCO)
2. **Tier 2**: Forcing Complex Core Banking/Custom-Model Agents into Low-Code Canvas Only (`48.0%` custom model/workflow flexibility, `$148,000/mo` TCO)
3. **Tier 3**: Disconnected Dual Silos (M365 Copilot + Separate Un-Integrated Foundry Web Portals) (`58.0%` adoption, `8.0 weeks` time-to-prod, `$162,000/mo` TCO)
4. **Tier 4**: 3-Ring Portfolio (`50%` `M365 Declarative Agents + Graph Connectors` + `30%` `Copilot Studio` + `20%` `Pro-Code Azure AI Foundry Custom Engine Agents` Federated in One UI Surface) (`94.6%` employee adoption, `100.0%` ACL fidelity, `2.4 weeks` time-to-prod, `100.0%` pro-code flexibility, `$64,000/mo` TCO)

## Usage

```bash
python experiments/enterprise_m365_copilot_extensibility_foundry_studio_pro_code_selection/run.py
python experiments/enterprise_m365_copilot_extensibility_foundry_studio_pro_code_selection/run.py --check
python -m unittest tests/test_enterprise_m365_copilot_extensibility_foundry_studio_pro_code_selection.py
```
