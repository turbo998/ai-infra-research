from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_semantic_cache_false_positive_prefix_kv_selection() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (
        root
        / "experiments"
        / "enterprise_semantic_cache_false_positive_cross_encoder_prefix_kv_selection"
        / "run.py"
    )
    subprocess.run([sys.executable, str(script)], check=True)
    results_path = script.parent / "results.json"
    data = json.loads(results_path.read_text(encoding="utf-8"))

    assert (
        data["experiment_id"]
        == "enterprise_semantic_cache_false_positive_cross_encoder_prefix_kv_selection"
    )
    assert data["summary_metrics"]["false_positive_reduction_factor"] == 177.5
    assert data["summary_metrics"]["gpu_saving_pct_vs_no_cache"] == 68.8
    archs = data["architectures"]
    assert len(archs) == 3
    assert archs[0]["false_positive_wrong_answer_pct"] == 14.2
    assert archs[2]["false_positive_wrong_answer_pct"] == 0.08
    assert archs[2]["required_h200_gpus_at_60rps"] == 10
