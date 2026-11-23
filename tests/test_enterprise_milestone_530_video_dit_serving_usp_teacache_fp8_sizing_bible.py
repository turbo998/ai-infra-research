from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_milestone_530_video_dit_serving_bible() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (
        root
        / "experiments"
        / "enterprise_milestone_530_video_dit_serving_usp_teacache_fp8_sizing_bible"
        / "run.py"
    )
    subprocess.run([sys.executable, str(script)], check=True)
    results_path = script.parent / "results.json"
    data = json.loads(results_path.read_text(encoding="utf-8"))

    assert (
        data["experiment_id"]
        == "enterprise_milestone_530_video_dit_serving_usp_teacache_fp8_sizing_bible"
    )
    assert data["summary_metrics"]["latency_speedup_factor_vs_single_gpu"] == 30.36
    assert data["summary_metrics"]["cost_reduction_pct_per_clip"] == 73.7
    archs = data["architectures"]
    assert len(archs) == 3
    assert archs[0]["clip_generation_seconds"] == 340.0
    assert archs[2]["clip_generation_seconds"] == 11.2
    assert archs[2]["cost_per_video_clip_usd"] == 0.077
