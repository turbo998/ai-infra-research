from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_k8s_aks_kaito_keda_coldstart_sizing() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (
        root
        / "experiments"
        / "enterprise_k8s_aks_kaito_keda_dragonfly_gds_coldstart_sizing_selection"
        / "run.py"
    )
    subprocess.run([sys.executable, str(script)], check=True)
    results_path = script.parent / "results.json"
    data = json.loads(results_path.read_text(encoding="utf-8"))

    assert (
        data["experiment_id"]
        == "enterprise_k8s_aks_kaito_keda_dragonfly_gds_coldstart_sizing_selection"
    )
    assert data["summary_metrics"]["cold_start_speedup_factor"] == 36.52
    assert data["summary_metrics"]["gpu_saving_pct_vs_legacy_hpa"] == 50.0
    archs = data["architectures"]
    assert len(archs) == 3
    assert archs[0]["pod_cold_start_seconds"] == 420.0
    assert archs[2]["pod_cold_start_seconds"] == 11.5
    assert archs[2]["total_h200_gpus_for_bursty_40rps"] == 16
