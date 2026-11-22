from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_multitenant_qos_wfq_selection() -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "experiments" / "enterprise_multitenant_qos_token_bucket_wfq_noisy_neighbor_isolation_selection" / "run.py"
    subprocess.run([sys.executable, str(script)], check=True)
    results_path = script.parent / "results.json"
    data = json.loads(results_path.read_text(encoding="utf-8"))

    assert data["experiment_id"] == "enterprise_multitenant_qos_token_bucket_wfq_noisy_neighbor_isolation_selection"
    tiers = data["tiers"]
    assert len(tiers) == 4
    assert tiers[0]["vip_p99_ttft_ms"] == 18400.0
    assert tiers[1]["cluster_gpu_utilization_pct"] == 46.5
    assert tiers[3]["vip_p99_ttft_ms"] == 325.0
    assert tiers[3]["cluster_gpu_utilization_pct"] == 88.5
