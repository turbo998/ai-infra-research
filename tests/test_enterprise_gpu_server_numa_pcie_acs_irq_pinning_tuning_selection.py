from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_gpu_server_numa_pcie_acs_tuning() -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "experiments" / "enterprise_gpu_server_numa_pcie_acs_irq_pinning_tuning_selection" / "run.py"
    subprocess.run([sys.executable, str(script)], check=True)
    results_path = script.parent / "results.json"
    data = json.loads(results_path.read_text(encoding="utf-8"))

    assert data["experiment_id"] == "enterprise_gpu_server_numa_pcie_acs_irq_pinning_tuning_selection"
    tiers = data["tiers"]
    assert len(tiers) == 4
    assert tiers[0]["gpudirect_rdma_busbw_gbps"] == 14.2
    assert tiers[0]["server_throughput_efficiency_pct"] == 72.0
    assert tiers[3]["gpudirect_rdma_busbw_gbps"] == 47.8
    assert tiers[3]["p99_tpot_ms"] == 17.8
    assert tiers[3]["server_throughput_efficiency_pct"] == 100.0
