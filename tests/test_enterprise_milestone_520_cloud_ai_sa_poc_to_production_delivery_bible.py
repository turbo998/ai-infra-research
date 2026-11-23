from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_milestone_520_cloud_ai_sa_delivery_bible() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (
        root
        / "experiments"
        / "enterprise_milestone_520_cloud_ai_sa_poc_to_production_delivery_bible"
        / "run.py"
    )
    subprocess.run([sys.executable, str(script)], check=True)
    results_path = script.parent / "results.json"
    data = json.loads(results_path.read_text(encoding="utf-8"))

    assert (
        data["experiment_id"]
        == "enterprise_milestone_520_cloud_ai_sa_poc_to_production_delivery_bible"
    )
    assert data["summary_metrics"]["total_gpu_reduction_pct"] == 87.5
    assert data["summary_metrics"]["token_cost_reduction_factor"] == 17.78
    stages = data["stages"]
    assert len(stages) == 3
    assert stages[0]["required_h200_gpus_at_100rps"] == 192
    assert stages[2]["required_h200_gpus_at_100rps"] == 24
    assert stages[2]["sla_goodput_pct"] == 99.4
