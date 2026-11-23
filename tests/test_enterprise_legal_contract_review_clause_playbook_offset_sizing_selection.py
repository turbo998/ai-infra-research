from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_legal_contract_review_clause_playbook_sizing() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (
        root
        / "experiments"
        / "enterprise_legal_contract_review_clause_playbook_offset_sizing_selection"
        / "run.py"
    )
    subprocess.run([sys.executable, str(script)], check=True)
    results_path = script.parent / "results.json"
    data = json.loads(results_path.read_text(encoding="utf-8"))

    assert (
        data["experiment_id"]
        == "enterprise_legal_contract_review_clause_playbook_offset_sizing_selection"
    )
    assert data["summary_metrics"]["review_speedup_factor_vs_128k"] == 8.29
    assert data["summary_metrics"]["gpu_saving_pct_vs_128k_stuffing"] == 75.0
    archs = data["architectures"]
    assert len(archs) == 3
    assert archs[0]["risk_clause_recall_pct"] == 80.6
    assert archs[2]["risk_clause_recall_pct"] == 98.4
    assert archs[2]["required_h200_gpus_for_5000_contracts_per_day"] == 8
