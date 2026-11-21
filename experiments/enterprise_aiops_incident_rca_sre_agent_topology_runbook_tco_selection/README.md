# Enterprise AIOps, Incident Triage & SRE Root-Cause Analysis (`RCA`) Agent Selection (`Azure SRE Agent` vs. `Datadog Bits AI` vs. `OTel Topology + Auto Read-Only Prefetch + Runbook Gate`)

Reproducible analytical model comparing four enterprise AIOps and incident response agent architectures across a 500-microservice production cluster (`alert noise reduction`, `RCA Top-1 accuracy`, `MTTR [minutes]`, and `annual TCO`).

- **Provenance**: `synthetic + anonymized_reported (independently_reproduced=false)`
- **Run**: `python experiments/enterprise_aiops_incident_rca_sre_agent_topology_runbook_tco_selection/run.py`
