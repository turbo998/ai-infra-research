# Enterprise ITSM (ServiceNow / Jira) & AIOps Incident Triage Agent Evaluator (Round 397)

Deterministic analytical evaluator comparing four enterprise ITSM helpdesk and AIOps incident triage architectures across `50,000` tickets/month:

1. **Tier 1 (`tier1_legacy_keyword_decision_tree`)**: Legacy Keyword Decision-Tree Bot (`16.0%` ticket automation rate, `240.0 min` [4.0 hours] avg MTTR, `$88,000/mo` total ops cost).
2. **Tier 2 (`tier2_passive_kb_rag_assistant`)**: Passive Knowledge-Base RAG Assistant (`42.0%` automation rate, `135.0 min` avg MTTR, `$54,000/mo`).
3. **Tier 3 (`tier3_unbounded_autonomous_shell_agent`)**: Unbounded Autonomous Remediation Agent (`68.0%` automation rate, `28.0 min` avg MTTR, `3.8%` unsafe production mutation incident rate, `$39,000/mo`).
4. **Tier 4 (`tier4_alert_dedup_cbr_readonly_diag_runbook_hitl`)**: Topology Alert Deduplication + Resolved-Ticket Case-Based Reasoning (`CBR`) + `100%` Automated Read-Only Diagnostics + Runbook-Gated Write Execution (`76.5%` automation rate [`+60.5 pts` vs. Tier 1], `14.5 min` avg MTTR [`16.6x` faster than Tier 1], `0.0%` unsafe production mutations, `$24,500/mo` [`-72.2%` cost vs. Tier 1]).

## Usage

```bash
python experiments/enterprise_itsm_servicedesk_aiops_incident_agent_selection/run.py
python experiments/enterprise_itsm_servicedesk_aiops_incident_agent_selection/run.py --check
```
