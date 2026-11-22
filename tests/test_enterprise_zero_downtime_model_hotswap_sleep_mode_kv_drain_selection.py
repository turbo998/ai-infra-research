from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_zero_downtime_model_hotswap_sleep_mode() -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "experiments" / "enterprise_zero_downtime_model_hotswap_sleep_mode_kv_drain_selection" / "run.py"
    subprocess.run([sys.executable, str(script)], check=True)
    results_path = script.parent / "results.json"
    data = json.loads(results_path.read_text(encoding="utf-8"))

    assert data["experiment_id"] == "enterprise_zero_downtime_model_hotswap_sleep_mode_kv_drain_selection"
    tiers = data["tiers"]
    assert len(tiers) == 4
    assert tiers[0]["switch_latency_sec"] == 185.0
    assert tiers[1]["extra_surge_gpus_pct"] == 100.0
    assert tiers[3]["switch_latency_sec"] == 1.8
    assert tiers[3]["dropped_active_requests_pct"] == 0.0
    assert tiers[3]["extra_surge_gpus_pct"] == 0.0
