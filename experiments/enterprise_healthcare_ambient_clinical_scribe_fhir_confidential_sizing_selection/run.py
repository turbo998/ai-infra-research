from __future__ import annotations

import json
from pathlib import Path


def build_results() -> dict:
    architectures = [
        {
            "arch_id": "unconstrained_generic_cloud_llm",
            "name": "Anti-Pattern 1: Generic Cloud API without Medical Lexicon or ICD-10 Grounding",
            "icd10_medication_coding_accuracy_pct": 74.5,
            "fhir_r4_schema_validity_pct": 89.2,
            "note_generation_p95_sec": 8.4,
            "required_onprem_h200_gpus_for_2000_clinicians": 0,
            "annual_tco_usd": 1440000.0,
            "action": "Send to a human",
        },
        {
            "arch_id": "onprem_unconstrained_monolith_70b",
            "name": "Baseline: On-Prem Unconstrained 70B LLM + Retry Loop (2,000 Clinicians)",
            "icd10_medication_coding_accuracy_pct": 82.0,
            "fhir_r4_schema_validity_pct": 94.0,
            "note_generation_p95_sec": 6.8,
            "required_onprem_h200_gpus_for_2000_clinicians": 16,
            "annual_tco_usd": 428000.0,
            "action": "Confirm",
        },
        {
            "arch_id": "confidential_medical_asr_slora_xgrammar_fhir",
            "name": "Optimal: On-Prem Confidential H200 + Medical Hotword ASR + Specialty S-LoRA + XGrammar FHIR + SNOMED/ICD-10 RAG",
            "icd10_medication_coding_accuracy_pct": 97.4,
            "fhir_r4_schema_validity_pct": 100.0,
            "note_generation_p95_sec": 1.6,
            "required_onprem_h200_gpus_for_2000_clinicians": 8,
            "annual_tco_usd": 214000.0,
            "action": "Act automatically",
        },
    ]

    return {
        "experiment_id": "enterprise_healthcare_ambient_clinical_scribe_fhir_confidential_sizing_selection",
        "title": "Enterprise Healthcare Ambient Clinical Scribe & HL7 FHIR R4 / ICD-10 Confidential AI Selection Guide (#521)",
        "provenance": "synthetic + anonymized_reported (independently_reproduced=false)",
        "summary_metrics": {
            "icd10_accuracy_gain_pts": round(97.4 - 74.5, 1),
            "note_latency_speedup_factor": round(8.4 / 1.6, 2),
            "tco_saving_pct_vs_saas_scribe": round((1.0 - 214000.0 / 1440000.0) * 100, 1),
        },
        "architectures": architectures,
    }


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    results = build_results()
    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("Wrote results.json")


if __name__ == "__main__":
    main()
