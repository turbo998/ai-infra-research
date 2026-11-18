# Enterprise Healthcare Revenue Cycle Management (`RCM`), `ICD-10 / CPT` Autonomous Medical Coding & Denial Appeals Agent (`R441`)

Deterministic evaluation suite comparing four hospital revenue cycle management (`RCM`), autonomous `ICD-10-CM / CPT` medical coding, inpatient `MS-DRG` Clinical Documentation Integrity (`CDI`), and `HL7 FHIR Da Vinci` prior-authorization/denial appeals architectures across a `2,800-bed` health system (`145,000` encounters/month, `155,000` active codes):

1. **Tier 1**: Manual Medical Coders + Legacy Encoder Lookup + Fax Prior-Auth (`12.8%` initial claim denial rate, `6.4%` NCCI/Excludes1 violations, `14.2%` missed CC/MCC DRG undercoding, `11.5` days DNFB turnaround, `$790,000/mo` write-off & labor cost)
2. **Tier 2**: Prompt-Only General LLM EHR Summary & ICD-10 Generator (`21.5%` denial rate, `19.8%` NCCI/Excludes1 violations, `9.5%` missed CC/MCC, `$960,000/mo` cost)
3. **Tier 3**: ICD-10 Vector RAG Without NCCI PTP / Excludes1 Rule Engine (`7.6%` denial rate, `5.2%` NCCI violations, `6.1%` missed CC/MCC, `1.2` days turnaround, `$360,000/mo` cost)
4. **Tier 4**: Assertion-Aware Clinical NLP (`NegEx/ConText`) + Deterministic `CMS NCCI PTP & Excludes1` AST Gate + Lab-Linked `MS-DRG CC/MCC` CDI Query + `HL7 FHIR Da Vinci` Prior-Auth Appeal Agent (`1.1%` denial rate, `0.0%` NCCI/Excludes1 violation, `0.4%` missed CC/MCC, `0.15` days / 3.6h turnaround, `$78,000/mo` TCO)

## Usage

```bash
python experiments/enterprise_healthcare_rcm_icd10_medical_coding_denial_appeals_agent_selection/run.py
python experiments/enterprise_healthcare_rcm_icd10_medical_coding_denial_appeals_agent_selection/run.py --check
python -m unittest tests/test_enterprise_healthcare_rcm_icd10_medical_coding_denial_appeals_agent_selection.py
```
