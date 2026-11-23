from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_codebase_rag_treesitter_ast_scip_sizing() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (
        root
        / "experiments"
        / "enterprise_codebase_rag_treesitter_ast_scip_callgraph_swe_sizing_selection"
        / "run.py"
    )
    subprocess.run([sys.executable, str(script)], check=True)
    results_path = script.parent / "results.json"
    data = json.loads(results_path.read_text(encoding="utf-8"))

    assert (
        data["experiment_id"]
        == "enterprise_codebase_rag_treesitter_ast_scip_callgraph_swe_sizing_selection"
    )
    assert data["summary_metrics"]["localization_recall_gain_pts"] == 48.1
    assert data["summary_metrics"]["gpu_saving_pct_vs_128k_stuffing"] == 83.3
    archs = data["architectures"]
    assert len(archs) == 3
    assert archs[0]["cross_file_bug_localization_recall_pct"] == 46.5
    assert archs[2]["cross_file_bug_localization_recall_pct"] == 94.6
    assert archs[2]["required_h200_gpus_for_2000_devs"] == 8
