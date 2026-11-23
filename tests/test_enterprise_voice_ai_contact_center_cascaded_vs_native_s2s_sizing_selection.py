from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_voice_ai_contact_center_cascaded_vs_native_s2s() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (
        root
        / "experiments"
        / "enterprise_voice_ai_contact_center_cascaded_vs_native_s2s_sizing_selection"
        / "run.py"
    )
    subprocess.run([sys.executable, str(script)], check=True)
    results_path = script.parent / "results.json"
    data = json.loads(results_path.read_text(encoding="utf-8"))

    assert (
        data["experiment_id"]
        == "enterprise_voice_ai_contact_center_cascaded_vs_native_s2s_sizing_selection"
    )
    assert data["summary_metrics"]["latency_speedup_vs_blocking_cascade"] == 9.27
    assert data["summary_metrics"]["tco_saving_pct_vs_native_realtime_api"] == 92.1
    archs = data["architectures"]
    assert len(archs) == 3
    assert archs[0]["p95_mouth_to_ear_ms"] == 2550.0
    assert archs[2]["p95_mouth_to_ear_ms"] == 275.0
    assert archs[2]["required_h200_gpus"] == 16
