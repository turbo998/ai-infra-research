from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_finserv_realtime_fraud_risk_cascade_sizing() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (
        root
        / "experiments"
        / "enterprise_finserv_realtime_fraud_risk_gbdt_slm_llm_cascade_sizing_selection"
        / "run.py"
    )
    subprocess.run([sys.executable, str(script)], check=True)
    results_path = script.parent / "results.json"
    data = json.loads(results_path.read_text(encoding="utf-8"))

    assert (
        data["experiment_id"]
        == "enterprise_finserv_realtime_fraud_risk_gbdt_slm_llm_cascade_sizing_selection"
    )
    assert data["summary_metrics"]["gpu_saving_pct_vs_inline_70b"] == 96.2
    assert data["summary_metrics"]["false_positive_reduction_pct_vs_xgboost"] == 73.5
    archs = data["architectures"]
    assert len(archs) == 3
    assert archs[0]["required_h200_gpus_at_5000tps"] == 320
    assert archs[2]["p99_inline_latency_ms"] == 24.8
    assert archs[2]["required_h200_gpus_at_5000tps"] == 12
