from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_nextgen_ai_factory_5yr_cascade_tco() -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "experiments" / "enterprise_nextgen_ai_factory_h200_gb200_rubin_5yr_cascade_tco_selection" / "run.py"
    subprocess.run([sys.executable, str(script)], check=True)
    results_path = script.parent / "results.json"
    data = json.loads(results_path.read_text(encoding="utf-8"))

    assert data["experiment_id"] == "enterprise_nextgen_ai_factory_h200_gb200_rubin_5yr_cascade_tco_selection"
    strategies = data["strategies"]
    assert len(strategies) == 4
    assert strategies[0]["five_year_total_tco_musd"] == 68.4
    assert strategies[3]["five_year_total_tco_musd"] == 36.8
    assert strategies[3]["blended_cost_per_1m_tokens_usd"] == 0.035
    assert strategies[3]["gpu_economic_lifespan_years"] == 5.0
