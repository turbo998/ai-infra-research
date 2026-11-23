from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_structured_output_xgrammar_llguidance_selection() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (
        root
        / "experiments"
        / "enterprise_structured_output_xgrammar_llguidance_constrained_decoding_selection"
        / "run.py"
    )
    subprocess.run([sys.executable, str(script)], check=True)
    results_path = script.parent / "results.json"
    data = json.loads(results_path.read_text(encoding="utf-8"))

    assert (
        data["experiment_id"]
        == "enterprise_structured_output_xgrammar_llguidance_constrained_decoding_selection"
    )
    assert data["summary_metrics"]["p99_e2e_speedup_vs_retry_loop"] == 5.73
    assert data["summary_metrics"]["p99_tpot_speedup_vs_legacy_fsm"] == 2.58
    assert data["summary_metrics"]["gpu_saving_pct_vs_retry_loop"] == 50.0
    engines = data["engines"]
    assert len(engines) == 3
    assert engines[0]["p99_e2e_latency_ms"] == 6420.0
    assert engines[2]["p99_e2e_latency_ms"] == 1120.0
    assert engines[2]["required_h200_gpus_at_50rps"] == 8
