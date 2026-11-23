from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_multi_lora_serving_slora_punica_sizing() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (
        root
        / "experiments"
        / "enterprise_multi_lora_serving_slora_punica_shared_base_sizing_selection"
        / "run.py"
    )
    subprocess.run([sys.executable, str(script)], check=True)
    results_path = script.parent / "results.json"
    data = json.loads(results_path.read_text(encoding="utf-8"))

    assert (
        data["experiment_id"]
        == "enterprise_multi_lora_serving_slora_punica_shared_base_sizing_selection"
    )
    assert data["summary_metrics"]["gpu_saving_pct_vs_dedicated_merged_pods"] == 93.8
    assert data["summary_metrics"]["annual_tco_saved_usd"] == 3216000
    archs = data["architectures"]
    assert len(archs) == 3
    assert archs[0]["required_h200_gpus"] == 128
    assert archs[2]["required_h200_gpus"] == 8
    assert archs[2]["p99_tpot_ms"] == 14.9
