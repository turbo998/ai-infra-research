from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_sovereign_airgapped_ai_cluster_delivery() -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "experiments" / "enterprise_sovereign_airgapped_ai_cluster_zarf_dragonfly_safetensors_selection" / "run.py"
    subprocess.run([sys.executable, str(script)], check=True)
    results_path = script.parent / "results.json"
    data = json.loads(results_path.read_text(encoding="utf-8"))

    assert data["experiment_id"] == "enterprise_sovereign_airgapped_ai_cluster_zarf_dragonfly_safetensors_selection"
    tiers = data["tiers"]
    assert len(tiers) == 4
    assert tiers[0]["delivery_lead_time_days"] == 21.5
    assert tiers[0]["rollout_64_nodes_minutes"] == 870.0
    assert tiers[3]["first_time_deploy_success_pct"] == 99.8
    assert tiers[3]["delivery_lead_time_days"] == 0.4
    assert tiers[3]["rollout_64_nodes_minutes"] == 3.2
