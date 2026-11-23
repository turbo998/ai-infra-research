from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_supply_chain_neuro_symbolic_timesfm_llm_milp() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (
        root
        / "experiments"
        / "enterprise_supply_chain_neuro_symbolic_timesfm_llm_milp_sizing_selection"
        / "run.py"
    )
    subprocess.run([sys.executable, str(script)], check=True)
    results_path = script.parent / "results.json"
    data = json.loads(results_path.read_text(encoding="utf-8"))

    assert (
        data["experiment_id"]
        == "enterprise_supply_chain_neuro_symbolic_timesfm_llm_milp_sizing_selection"
    )
    assert data["summary_metrics"]["stockout_reduction_pct_vs_legacy_erp"] == 70.3
    assert data["summary_metrics"]["gpu_saving_pct_vs_pure_llm"] == 87.5
    archs = data["architectures"]
    assert len(archs) == 3
    assert archs[0]["constraint_satisfaction_rate_pct"] == 61.5
    assert archs[2]["constraint_satisfaction_rate_pct"] == 100.0
    assert archs[2]["required_h200_gpus"] == 8
