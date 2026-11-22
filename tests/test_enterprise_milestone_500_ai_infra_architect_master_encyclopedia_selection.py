from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_milestone_500_master_encyclopedia() -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "experiments" / "enterprise_milestone_500_ai_infra_architect_master_encyclopedia_selection" / "run.py"
    subprocess.run([sys.executable, str(script)], check=True)
    results_path = script.parent / "results.json"
    data = json.loads(results_path.read_text(encoding="utf-8"))

    assert data["experiment_id"] == "enterprise_milestone_500_ai_infra_architect_master_encyclopedia_selection"
    assert data["milestone_500_summary"]["total_bilingual_playbooks"] == 500
    levels = data["levels"]
    assert len(levels) == 4
    assert levels[0]["blended_cost_per_1m_tokens_usd"] == 0.680
    assert levels[3]["blended_cost_per_1m_tokens_usd"] == 0.028
    assert levels[3]["cluster_effective_goodput_pct"] == 91.2
    assert levels[3]["three_year_enterprise_tco_musd"] == 5.8
