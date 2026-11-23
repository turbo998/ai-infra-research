# Enterprise Multi-Region Active-Active LLM Disaster Recovery (`99.995% SLA · 0.85s RTO`) Sizer (`#516`)

Reproducible analytical model comparing **Single-Region Client Retry**, **Active-Passive `100% + 100%` Standby DR**, and **Active-Active `55% + 55%` Dual-Region PTU/GPU + `800ms` TTFT Circuit Breaker + Cross-Cloud PAYGO Spillover**.

## Run

```bash
python experiments/enterprise_multiregion_active_active_llm_dr_ptu_failover_sizing_selection/run.py
```

- **Provenance**: `synthetic + anonymized_reported (independently_reproduced=false)`
- **Bilingual Guides**:
  - [Chinese FAQ](../../docs/components/enterprise-multiregion-active-active-llm-dr-ptu-failover-sizing-selection-faq.md)
  - [English FAQ](../../docs/components/enterprise-multiregion-active-active-llm-dr-ptu-failover-sizing-selection-faq.en.md)
- **Interactive H5 Lab**: [docs/h5/multiregion-active-active-llm-dr-ptu-failover-sizer-lab.html](../../docs/h5/multiregion-active-active-llm-dr-ptu-failover-sizer-lab.html)
