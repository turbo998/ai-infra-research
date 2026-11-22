from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_browser_computer_use_agent_sizing() -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "experiments" / "enterprise_browser_computer_use_agent_axtree_som_vlm_sizing_selection" / "run.py"
    subprocess.run([sys.executable, str(script)], check=True)
    results_path = script.parent / "results.json"
    data = json.loads(results_path.read_text(encoding="utf-8"))

    assert data["experiment_id"] == "enterprise_browser_computer_use_agent_axtree_som_vlm_sizing_selection"
    tiers = data["tiers"]
    assert len(tiers) == 4
    assert tiers[0]["required_h200_gpus_500_sessions"] == 64
    assert tiers[3]["workflow_success_rate_pct"] == 95.6
    assert tiers[3]["workflow_e2el_sec"] == 7.8
    assert tiers[3]["required_h200_gpus_500_sessions"] == 6
