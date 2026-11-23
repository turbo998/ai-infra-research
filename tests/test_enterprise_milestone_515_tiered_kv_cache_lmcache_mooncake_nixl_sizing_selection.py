from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_milestone_515_tiered_kv_cache_lmcache_mooncake_nixl() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (
        root
        / "experiments"
        / "enterprise_milestone_515_tiered_kv_cache_lmcache_mooncake_nixl_sizing_selection"
        / "run.py"
    )
    subprocess.run([sys.executable, str(script)], check=True)
    results_path = script.parent / "results.json"
    data = json.loads(results_path.read_text(encoding="utf-8"))

    assert (
        data["experiment_id"]
        == "enterprise_milestone_515_tiered_kv_cache_lmcache_mooncake_nixl_sizing_selection"
    )
    assert data["summary_metrics"]["resume_ttft_speedup_factor_vs_recompute"] == 15.59
    assert data["summary_metrics"]["gpu_saving_pct_vs_hbm_only"] == 66.7
    tiers = data["tiers"]
    assert len(tiers) == 3
    assert tiers[0]["turn_resume_p95_ttft_ms"] == 1840.0
    assert tiers[2]["turn_resume_p95_ttft_ms"] == 118.0
    assert tiers[2]["required_h200_gpus"] == 12
