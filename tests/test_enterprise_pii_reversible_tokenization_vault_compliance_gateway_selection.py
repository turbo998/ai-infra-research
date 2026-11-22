from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_pii_reversible_tokenization_gateway() -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "experiments" / "enterprise_pii_reversible_tokenization_vault_compliance_gateway_selection" / "run.py"
    subprocess.run([sys.executable, str(script)], check=True)
    results_path = script.parent / "results.json"
    data = json.loads(results_path.read_text(encoding="utf-8"))

    assert data["experiment_id"] == "enterprise_pii_reversible_tokenization_vault_compliance_gateway_selection"
    tiers = data["tiers"]
    assert len(tiers) == 4
    assert tiers[0]["downstream_llm_task_accuracy_pct"] == 52.0
    assert tiers[1]["gateway_overhead_ms"] == 385.0
    assert tiers[3]["pii_detection_recall_pct"] == 98.6
    assert tiers[3]["downstream_llm_task_accuracy_pct"] == 94.2
    assert tiers[3]["gateway_overhead_ms"] == 4.5
