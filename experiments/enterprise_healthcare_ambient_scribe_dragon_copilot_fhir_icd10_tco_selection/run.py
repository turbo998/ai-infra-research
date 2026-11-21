from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_results() -> dict:
    physicians = 2000
    tiers = [
        {
            "tier": "Tier 1: Generic Cloud ASR + Free-Text 70B SOAP Note + Zero-Shot ICD-10 Guessing",
            "dosage_or_negation_error_rate_pct": 8.6,
            "icd10_coding_compliance_pct": 70.6,
            "physician_charting_sec_per_visit": 180,
            "annual_platform_tco_usd": 960000,
        },
        {
            "tier": "Tier 2: Turnkey Clinical SaaS Flagship (Microsoft Dragon Copilot [Nuance DAX] / Native Epic)",
            "dosage_or_negation_error_rate_pct": 0.15,
            "icd10_coding_compliance_pct": 97.8,
            "physician_charting_sec_per_visit": 48,
            "annual_platform_tco_usd": 7200000,
        },
        {
            "tier": "Tier 3: Cloud Healthcare API (AWS HealthScribe / Azure AI Health Insights Per-Minute API)",
            "dosage_or_negation_error_rate_pct": 0.25,
            "icd10_coding_compliance_pct": 96.4,
            "physician_charting_sec_per_visit": 55,
            "annual_platform_tco_usd": 2880000,
        },
        {
            "tier": "Tier 4: On-Prem Medical Hotword ASR + RxNorm Dosage Gate + HL7 FHIR R4 + ICD-10 Tree + Audio Citation",
            "dosage_or_negation_error_rate_pct": 0.0,
            "icd10_coding_compliance_pct": 98.4,
            "physician_charting_sec_per_visit": 45,
            "annual_platform_tco_usd": 380000,
        },
    ]
    return {
        "experiment_id": "enterprise_healthcare_ambient_scribe_dragon_copilot_fhir_icd10_tco_selection",
        "Provenance": "synthetic + anonymized_reported (independently_reproduced=false)",
        "workload_baseline": {
            "attending_physicians": physicians,
            "patient_encounters_per_physician_day": 24,
            "avg_encounter_audio_minutes": 10,
        },
        "tiers": tiers,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    out_path = Path(__file__).resolve().parent / "results.json"
    payload = build_results()

    if args.check:
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        assert existing == payload, "results.json drift detected"
        print("OK: enterprise_healthcare_ambient_scribe_dragon_copilot_fhir_icd10_tco_selection verified.")
        return

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Wrote results.json")


if __name__ == "__main__":
    main()
