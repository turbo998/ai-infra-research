# Enterprise HR Talent Acquisition, ESCO/O*NET Skill Ontology, EEOC/EU AI Act Bias Audit & RBAC Policy Copilot Evaluator

Deterministic evaluator comparing four enterprise HR talent acquisition and employee policy copilot architectures across 45,000 global employees, 180,000 monthly candidate resumes, and 60,000 monthly HR policy queries:
- **Tier 1**: Legacy ATS Exact Keyword Boolean Filter + Manual Recruiter Screening (`58.0%` shortlist `NDCG@10`, `6.5%` EEOC adverse-impact violation rate, `4.2%` policy leak rate, `$152,000/mo`)
- **Tier 2**: Unmasked Raw Resume Fed Directly to Holistic 1–100 LLM Scorer (`74.5%` `NDCG@10`, `24.8%` EEOC adverse-impact violation rate, `12.6%` policy leak rate, `$215,000/mo`)
- **Tier 3**: PII-Redacted Resume + Single Holistic Prompt Scoring + Basic HR FAQ RAG (`81.2%` `NDCG@10`, `7.4%` EEOC violation rate, `3.1%` policy leak rate, `$68,000/mo`)
- **Tier 4**: PII/Proxy-Blind Parser + `ESCO / O*NET` Skill Ontology + Evidence-Linked Binary Competency Rubric + `EEOC 4/5ths Rule` Gate + `Workday / SuccessFactors` RBAC Policy Copilot (`93.6%` `NDCG@10`, `0.0%` EEOC violation rate, `0.0%` policy leak rate, `$23,000/mo`)

## Usage
```bash
python experiments/enterprise_hr_talent_recruitment_policy_copilot_bias_audit_selection/run.py
python experiments/enterprise_hr_talent_recruitment_policy_copilot_bias_audit_selection/run.py --check
```
