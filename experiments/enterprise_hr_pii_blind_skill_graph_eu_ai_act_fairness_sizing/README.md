# Enterprise HR Talent Acquisition & EU AI Act Bias Auditing (`500K Resumes/Month`): `Raw 70B Ranking` vs. `Keyword Boolean ATS` vs. `PII-Blind ESCO Skill-Graph + Auditable Rubric + 4/5ths Fairness Gate` (`#544`)

- **Chinese FAQ**: [`../../docs/components/enterprise-hr-pii-blind-skill-graph-eu-ai-act-fairness-sizing-faq.md`](../../docs/components/enterprise-hr-pii-blind-skill-graph-eu-ai-act-fairness-sizing-faq.md)
- **English FAQ**: [`../../docs/components/enterprise-hr-pii-blind-skill-graph-eu-ai-act-fairness-sizing-faq.en.md`](../../docs/components/enterprise-hr-pii-blind-skill-graph-eu-ai-act-fairness-sizing-faq.en.md)
- **Interactive H5 Lab**: [`../../docs/h5/hr-pii-blind-skill-graph-eu-ai-act-fairness-lab.html`](../../docs/h5/hr-pii-blind-skill-graph-eu-ai-act-fairness-lab.html)
- **Interactive Sizer**: [`../../docs/reports/enterprise-hr-pii-blind-skill-graph-eu-ai-act-fairness-sizing-sizer.html`](../../docs/reports/enterprise-hr-pii-blind-skill-graph-eu-ai-act-fairness-sizing-sizer.html)
- **7-Card Report**: [`../../docs/reports/enterprise-hr-pii-blind-skill-graph-eu-ai-act-fairness-sizing-cards.html`](../../docs/reports/enterprise-hr-pii-blind-skill-graph-eu-ai-act-fairness-sizing-cards.html)

## Reproduce
```bash
python experiments/enterprise_hr_pii_blind_skill_graph_eu_ai_act_fairness_sizing/run.py
python -m pytest tests/test_enterprise_hr_pii_blind_skill_graph_eu_ai_act_fairness_sizing.py -q
```
