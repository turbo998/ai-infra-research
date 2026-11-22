from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_agentic_memory_bitemporal_graph_sizing() -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "experiments" / "enterprise_agentic_memory_mem0_zep_bitemporal_graph_kv_sizing_selection" / "run.py"
    subprocess.run([sys.executable, str(script)], check=True)
    results_path = script.parent / "results.json"
    data = json.loads(results_path.read_text(encoding="utf-8"))

    assert data["experiment_id"] == "enterprise_agentic_memory_mem0_zep_bitemporal_graph_kv_sizing_selection"
    tiers = data["tiers"]
    assert len(tiers) == 4
    assert tiers[0]["required_h200_gpus_at_50rps"] == 80
    assert tiers[2]["temporal_state_accuracy_pct"] == 54.5
    assert tiers[3]["temporal_state_accuracy_pct"] == 94.8
    assert tiers[3]["prompt_tokens_per_turn"] == 1850
    assert tiers[3]["required_h200_gpus_at_50rps"] == 4
