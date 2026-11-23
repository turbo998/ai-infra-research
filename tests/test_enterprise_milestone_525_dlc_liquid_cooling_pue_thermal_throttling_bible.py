from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_milestone_525_dlc_liquid_cooling_bible() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (
        root
        / "experiments"
        / "enterprise_milestone_525_dlc_liquid_cooling_pue_thermal_throttling_bible"
        / "run.py"
    )
    subprocess.run([sys.executable, str(script)], check=True)
    results_path = script.parent / "results.json"
    data = json.loads(results_path.read_text(encoding="utf-8"))

    assert (
        data["experiment_id"]
        == "enterprise_milestone_525_dlc_liquid_cooling_pue_thermal_throttling_bible"
    )
    assert data["summary_metrics"]["pue_improvement"] == 0.42
    assert data["summary_metrics"]["three_year_power_tco_saving_pct"] == 45.4
    opts = data["cooling_options"]
    assert len(opts) == 3
    assert opts[0]["gpu_thermal_throttle_loss_pct"] == 18.5
    assert opts[2]["facility_pue"] == 1.12
    assert opts[2]["racks_required_for_256_gpus"] == 8
