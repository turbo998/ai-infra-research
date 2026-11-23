from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_durable_agent_workflow_temporal_langgraph_sizing() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (
        root
        / "experiments"
        / "enterprise_durable_agent_workflow_temporal_langgraph_hitl_sizing_selection"
        / "run.py"
    )
    subprocess.run([sys.executable, str(script)], check=True)
    results_path = script.parent / "results.json"
    data = json.loads(results_path.read_text(encoding="utf-8"))

    assert (
        data["experiment_id"]
        == "enterprise_durable_agent_workflow_temporal_langgraph_hitl_sizing_selection"
    )
    assert data["summary_metrics"]["completion_rate_gain_pts"] == 28.5
    assert data["summary_metrics"]["gpu_saving_pct_vs_in_memory_loop"] == 50.0
    archs = data["architectures"]
    assert len(archs) == 3
    assert archs[0]["long_horizon_20step_completion_rate_pct"] == 71.2
    assert archs[2]["long_horizon_20step_completion_rate_pct"] == 99.7
    assert archs[2]["required_h200_gpus_for_1000_workflows_per_hr"] == 12
