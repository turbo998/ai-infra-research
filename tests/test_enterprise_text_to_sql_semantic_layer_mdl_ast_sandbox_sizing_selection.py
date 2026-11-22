from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_text_to_sql_semantic_layer_mdl_sizing() -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "experiments" / "enterprise_text_to_sql_semantic_layer_mdl_ast_sandbox_sizing_selection" / "run.py"
    subprocess.run([sys.executable, str(script)], check=True)
    results_path = script.parent / "results.json"
    data = json.loads(results_path.read_text(encoding="utf-8"))

    assert data["experiment_id"] == "enterprise_text_to_sql_semantic_layer_mdl_ast_sandbox_sizing_selection"
    tiers = data["tiers"]
    assert len(tiers) == 4
    assert tiers[0]["business_sql_accuracy_pct"] == 38.5
    assert tiers[0]["required_h200_gpus_at_30qps"] == 48
    assert tiers[3]["business_sql_accuracy_pct"] == 95.4
    assert tiers[3]["prompt_tokens_per_query"] == 2400
    assert tiers[3]["required_h200_gpus_at_30qps"] == 4
