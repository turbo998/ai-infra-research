# Enterprise Biopharma Clinical Trial Eligibility Matching, Pharmacovigilance ICSR MedDRA Coding & FDA 21 CFR Part 11 Agent Evaluator

Deterministic evaluator comparing four biopharma clinical trial recruitment and pharmacovigilance (`ICSR / MedDRA`) architectures across 85 active Phase II/III trials, 120,000 monthly patient EHR profiles, and 25,000 monthly adverse event cases:
- **Tier 1**: Manual CRA EHR Chart Review + Manual Pharmacovigilance `MedDRA / E2B(R3)` Coding (`88.5%` eligibility F1, `3.2%` exclusion miss rate, `91.0%` MedDRA PT accuracy, `$260,000/mo`)
- **Tier 2**: Monolithic Prompt LLM Evaluating 40 Criteria & Free-Form MedDRA Terms (`64.2%` eligibility F1, `16.8%` exclusion safety miss rate, `71.5%` MedDRA PT accuracy, `$310,000/mo`)
- **Tier 3**: Standard Medical Note Vector RAG + Unconstrained LLM Criteria Check (`78.4%` eligibility F1, `6.4%` exclusion miss rate, `84.8%` MedDRA PT accuracy, `$115,000/mo`)
- **Tier 4**: Atomic Criterion Logic Compiler (`FHIR Lab/Date SQL + Clinical NLP`) + Constrained `MedDRA v27` Hierarchy Resolver + `FDA 21 CFR Part 11` Audit-Trail HITL Agent (`96.8%` eligibility F1, `0.0%` exclusion safety miss rate, `98.4%` MedDRA PT accuracy, `$38,000/mo`)

## Usage
```bash
python experiments/enterprise_pharma_clinical_trial_matching_pharmacovigilance_agent_selection/run.py
python experiments/enterprise_pharma_clinical_trial_matching_pharmacovigilance_agent_selection/run.py --check
```
