from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_graphrag_lazygraphrag_hybrid_sizing() -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "experiments" / "enterprise_graphrag_lazygraphrag_hybrid_reranker_sizing_selection" / "run.py"
    subprocess.run([sys.executable, str(script)], check=True)
    results_path = script.parent / "results.json"
    data = json.loads(results_path.read_text(encoding="utf-8"))

    assert data["experiment_id"] == "enterprise_graphrag_lazygraphrag_hybrid_reranker_sizing_selection"
    tiers = data["tiers"]
    assert len(tiers) == 4
    assert tiers[1]["index_build_cost_500m_tokens_usd"] == 4500
    assert tiers[3]["index_build_cost_500m_tokens_usd"] == 128
    assert tiers[3]["factoid_accuracy_pct"] == 94.8
    assert tiers[3]["global_multihop_accuracy_pct"] == 93.2
    assert tiers[3]["required_h200_gpus_at_50qps"] == 6
