from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_llm_eval_cicd_judge_calibration() -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "experiments" / "enterprise_llm_eval_cicd_judge_calibration_shadow_canary_selection" / "run.py"
    subprocess.run([sys.executable, str(script)], check=True)
    results_path = script.parent / "results.json"
    data = json.loads(results_path.read_text(encoding="utf-8"))

    assert data["experiment_id"] == "enterprise_llm_eval_cicd_judge_calibration_shadow_canary_selection"
    tiers = data["tiers"]
    assert len(tiers) == 4
    assert tiers[1]["human_agreement_kappa"] == 0.51
    assert tiers[3]["regression_catch_rate_pct"] == 98.4
    assert tiers[3]["human_agreement_kappa"] == 0.89
    assert tiers[3]["ci_gate_latency_minutes"] == 2.8
