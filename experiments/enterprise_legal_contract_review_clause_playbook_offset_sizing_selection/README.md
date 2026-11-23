# Enterprise LegalTech Contract Review & Clause Redlining (`5,000 Contracts/Day`) Sizer (`#531`)

Reproducible analytical model comparing **128K Full-Contract Stuffing (`80.6%` risk recall due to Lost-in-the-Middle, `11.4%` hallucinated citations, `68.0s`, `32 GPUs` ✕)**, **Serial Clause-by-Clause 70B Review (`95.0s`)**, and **Parallel Clause DAG + Playbook `SGLang RadixAttention` + `8B` Standard-Clause Filter / `70B` Risk Redliner + Exact Character-Offset Word/DOCX Grounding (`98.4%` recall, `0.0%` hallucinated quotes, `8.2s` P95 [`8.29x` faster], `8x H200 GPUs`, `-75.0%` GPUs 🏆)**.

## Run

```bash
python experiments/enterprise_legal_contract_review_clause_playbook_offset_sizing_selection/run.py
```

- **Provenance**: `synthetic + anonymized_reported (independently_reproduced=false)`
- **Bilingual Guides**:
  - [Chinese FAQ](../../docs/components/enterprise-legal-contract-review-clause-playbook-offset-sizing-selection-faq.md)
  - [English FAQ](../../docs/components/enterprise-legal-contract-review-clause-playbook-offset-sizing-selection-faq.en.md)
- **Interactive H5 Lab**: [docs/h5/legal-contract-review-clause-playbook-offset-lab.html](../../docs/h5/legal-contract-review-clause-playbook-offset-lab.html)
