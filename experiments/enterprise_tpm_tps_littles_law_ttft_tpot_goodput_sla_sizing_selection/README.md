# Enterprise `TPM / TPS / PTU` Conversion, `Little's Law` Concurrency & `TTFT / TPOT / ITL / Goodput` SLA Sizing Model (`#448`)

Reproducible analytical model comparing naive `1:1` raw TPM sizing, 100% peak PTU over-provisioning, Little's Law active-slot sizing, and `P80` Prefix-Deducted PTU/GPU + `P20` Serverless Burst with Dual-SLA Goodput gating across 5 enterprise scenarios.

- **Provenance**: `synthetic + anonymized_reported (independently_reproduced=false)`
- **Run**: `python experiments/enterprise_tpm_tps_littles_law_ttft_tpot_goodput_sla_sizing_selection/run.py --check`
