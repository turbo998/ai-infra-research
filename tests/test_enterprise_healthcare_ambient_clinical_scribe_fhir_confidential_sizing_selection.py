from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_healthcare_ambient_clinical_scribe_fhir_sizing() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (
        root
        / "experiments"
        / "enterprise_healthcare_ambient_clinical_scribe_fhir_confidential_sizing_selection"
        / "run.py"
    )
    subprocess.run([sys.executable, str(script)], check=True)
    results_path = script.parent / "results.json"
    data = json.loads(results_path.read_text(encoding="utf-8"))

    assert (
        data["experiment_id"]
        == "enterprise_healthcare_ambient_clinical_scribe_fhir_confidential_sizing_selection"
    )
    assert data["summary_metrics"]["icd10_accuracy_gain_pts"] == 22.9
    assert data["summary_metrics"]["tco_saving_pct_vs_saas_scribe"] == 85.1
    archs = data["architectures"]
    assert len(archs) == 3
    assert archs[2]["icd10_medication_coding_accuracy_pct"] == 97.4
    assert archs[2]["required_onprem_h200_gpus_for_2000_clinicians"] == 8
