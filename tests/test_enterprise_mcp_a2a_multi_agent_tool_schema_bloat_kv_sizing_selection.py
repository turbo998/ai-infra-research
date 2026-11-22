from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_mcp_a2a_multi_agent_kv_sizing() -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "experiments" / "enterprise_mcp_a2a_multi_agent_tool_schema_bloat_kv_sizing_selection" / "run.py"
    subprocess.run([sys.executable, str(script)], check=True)
    results_path = script.parent / "results.json"
    data = json.loads(results_path.read_text(encoding="utf-8"))

    assert data["experiment_id"] == "enterprise_mcp_a2a_multi_agent_tool_schema_bloat_kv_sizing_selection"
    tiers = data["tiers"]
    assert len(tiers) == 4
    assert tiers[0]["tokens_per_agent_task"] == 245000
    assert tiers[0]["required_h200_gpus_at_20tps"] == 48
    assert tiers[3]["tokens_per_agent_task"] == 26500
    assert tiers[3]["tool_selection_accuracy_pct"] == 94.6
    assert tiers[3]["required_h200_gpus_at_20tps"] == 6
