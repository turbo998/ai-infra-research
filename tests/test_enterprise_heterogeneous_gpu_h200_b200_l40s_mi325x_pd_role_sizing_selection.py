from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_heterogeneous_gpu_pd_role_sizing() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (
        root
        / "experiments"
        / "enterprise_heterogeneous_gpu_h200_b200_l40s_mi325x_pd_role_sizing_selection"
        / "run.py"
    )
    subprocess.run([sys.executable, str(script)], check=True)
    results_path = script.parent / "results.json"
    data = json.loads(results_path.read_text(encoding="utf-8"))

    assert (
        data["experiment_id"]
        == "enterprise_heterogeneous_gpu_h200_b200_l40s_mi325x_pd_role_sizing_selection"
    )
    assert data["summary_metrics"]["three_year_tco_saving_pct_vs_all_h200"] == 44.8
    assert data["summary_metrics"]["decode_tpot_speedup_vs_misaligned_l40s"] == 4.37
    strategies = data["strategies"]
    assert len(strategies) == 3
    assert strategies[0]["p99_tpot_ms"] == 62.0
    assert strategies[2]["p99_tpot_ms"] == 14.2
    assert strategies[2]["three_year_tco_usd"] == 1418000.0
