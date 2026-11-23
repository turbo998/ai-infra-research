from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_paygo_ptu_selfhosted_k8s_tco_ai_gateway() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (
        root
        / "experiments"
        / "enterprise_paygo_ptu_selfhosted_k8s_tco_ai_gateway_selection"
        / "run.py"
    )
    subprocess.run([sys.executable, str(script)], check=True)
    results_path = script.parent / "results.json"
    data = json.loads(results_path.read_text(encoding="utf-8"))

    assert (
        data["experiment_id"]
        == "enterprise_paygo_ptu_selfhosted_k8s_tco_ai_gateway_selection"
    )
    assert data["summary_metrics"]["ptu_spillover_saving_vs_pure_paygo_pct"] == 39.6
    assert data["summary_metrics"]["self_hosted_cost_reduction_factor_vs_paygo"] == 9.23
    models = data["deployment_models"]
    assert len(models) == 4
    assert models[0]["blended_cost_per_1m_tokens_usd"] == 2.85
    assert models[3]["blended_cost_per_1m_tokens_usd"] == 0.052
