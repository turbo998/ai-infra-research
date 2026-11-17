# Enterprise Healthcare Clinical Documentation (Ambient AI Scribe), FHIR EHR & ICD-10 Coding Evaluator (Round 406)

Deterministic evaluator comparing four healthcare ambient clinical scribe and automated medical coding (`HL7 FHIR R4 / ICD-10-CM / CPT`) architectures across 500,000 monthly patient consultations (7,500,000 audio minutes/month):
- **Tier 1**: Generic Cloud ASR + Unconstrained Free-Text SOAP Note Prompt (`84.5%` medication/negation accuracy, `61.2%` ICD-10 coding pass rate, `6.8 min` physician review, `$62,000/mo`)
- **Tier 2**: End-to-End Raw Audio Direct to Frontier Multimodal Audio-LLM (`91.0%` medication/negation accuracy, `74.5%` ICD-10 pass rate, `4.5 min` physician review, `$158,000/mo`)
- **Tier 3**: Medical-Tuned ASR + `70B` LLM SOAP Generation (Unlinked to RxNorm / ICD-10 Tree) (`96.2%` medication/negation accuracy, `83.8%` ICD-10 pass rate, `3.1 min` physician review, `$44,000/mo`)
- **Tier 4**: Medical Hotword ASR + RxNorm Dosage Safety Gate + HL7 FHIR R4 Constrained Schema + Two-Stage ICD-10/CPT Ontology Tree Search + Utterance Citation Links (`99.7%` medication/negation accuracy, `97.8%` ICD-10 pass rate, `0.9 min [54s]` physician review, `$14,200/mo`)

## Usage
```bash
python experiments/enterprise_healthcare_clinical_scribe_fhir_selection/run.py
python experiments/enterprise_healthcare_clinical_scribe_fhir_selection/run.py --check
```
