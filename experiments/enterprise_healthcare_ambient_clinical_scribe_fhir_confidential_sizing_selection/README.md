# Enterprise Healthcare Ambient Clinical Scribe & `HL7 FHIR R4 / ICD-10` Confidential AI Sizer (`#521`)

Reproducible analytical model evaluating **Ambient Clinical Documentation (`2,000` clinicians)** across Generic Cloud APIs, On-Prem Unconstrained 70B LLMs, and **On-Prem Confidential `H200 PPCIE` + Medical Hotword ASR + 24-Specialty `S-LoRA` + `XGrammar HL7 FHIR R4` + `SNOMED/ICD-10` Deterministic Grounding (`97.4%` coding accuracy, `100.0%` FHIR validity, `1.6s` P95 latency, `8x H200 GPUs`, `-85.1%` TCO vs. SaaS)**.

## Run

```bash
python experiments/enterprise_healthcare_ambient_clinical_scribe_fhir_confidential_sizing_selection/run.py
```

- **Provenance**: `synthetic + anonymized_reported (independently_reproduced=false)`
- **Bilingual Guides**:
  - [Chinese FAQ](../../docs/components/enterprise-healthcare-ambient-clinical-scribe-fhir-confidential-sizing-selection-faq.md)
  - [English FAQ](../../docs/components/enterprise-healthcare-ambient-clinical-scribe-fhir-confidential-sizing-selection-faq.en.md)
- **Interactive H5 Lab**: [docs/h5/healthcare-ambient-clinical-scribe-fhir-confidential-lab.html](../../docs/h5/healthcare-ambient-clinical-scribe-fhir-confidential-lab.html)
