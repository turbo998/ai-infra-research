from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_rlvr_grpo_verl_hybridengine_sizing() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (
        root
        / "experiments"
        / "enterprise_rlvr_grpo_verl_hybridengine_rollout_resharding_sizing_selection"
        / "run.py"
    )
    subprocess.run([sys.executable, str(script)], check=True)
    results_path = script.parent / "results.json"
    data = json.loads(results_path.read_text(encoding="utf-8"))

    assert (
        data["experiment_id"]
        == "enterprise_rlvr_grpo_verl_hybridengine_rollout_resharding_sizing_selection"
    )
    assert data["summary_metrics"]["throughput_speedup_factor"] == 2.45
    assert data["summary_metrics"]["gpu_saving_pct_vs_static_split"] == 62.5
    archs = data["architectures"]
    assert len(archs) == 3
    assert archs[0]["gpu_idle_bubble_pct"] == 54.0
    assert archs[2]["gpu_idle_bubble_pct"] == 8.4
    assert archs[2]["required_h200_gpus_for_7day_grpo"] == 24
