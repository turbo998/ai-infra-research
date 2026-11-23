from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_multiregion_active_active_llm_dr_ptu_failover() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (
        root
        / "experiments"
        / "enterprise_multiregion_active_active_llm_dr_ptu_failover_sizing_selection"
        / "run.py"
    )
    subprocess.run([sys.executable, str(script)], check=True)
    results_path = script.parent / "results.json"
    data = json.loads(results_path.read_text(encoding="utf-8"))

    assert (
        data["experiment_id"]
        == "enterprise_multiregion_active_active_llm_dr_ptu_failover_sizing_selection"
    )
    assert data["summary_metrics"]["rto_speedup_factor_vs_single_region"] == 52.94
    assert data["summary_metrics"]["tco_saving_pct_vs_active_passive"] == 42.5
    archs = data["dr_architectures"]
    assert len(archs) == 3
    assert archs[0]["failover_rto_seconds"] == 45.0
    assert archs[2]["failover_rto_seconds"] == 0.85
    assert archs[2]["annual_platform_tco_usd"] == 276000.0
