from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_2026_ai_infra_breakthrough_radar() -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "experiments" / "enterprise_2026_ai_infra_breakthrough_radar_dynamo_deepep_vllmv1_sglang_selection" / "run.py"
    subprocess.run([sys.executable, str(script)], check=True)
    results_path = script.parent / "results.json"
    data = json.loads(results_path.read_text(encoding="utf-8"))

    assert data["experiment_id"] == "enterprise_2026_ai_infra_breakthrough_radar_dynamo_deepep_vllmv1_sglang_selection"
    gens = data["generations"]
    assert len(gens) == 4
    assert gens[0]["output_tokens_per_sec_per_gpu"] == 620.0
    assert gens[0]["required_h200_gpus_at_50rps"] == 48
    assert gens[3]["output_tokens_per_sec_per_gpu"] == 4150.0
    assert gens[3]["p99_tpot_ms"] == 12.4
    assert gens[3]["required_h200_gpus_at_50rps"] == 8
