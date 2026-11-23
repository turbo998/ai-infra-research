from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_milestone_510_nvlink_infiniband_rocev2_gpudirect_busbw_bible() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (
        root
        / "experiments"
        / "enterprise_milestone_510_nvlink_infiniband_rocev2_gpudirect_busbw_bible"
        / "run.py"
    )
    subprocess.run([sys.executable, str(script)], check=True)
    results_path = script.parent / "results.json"
    data = json.loads(results_path.read_text(encoding="utf-8"))

    assert (
        data["experiment_id"]
        == "enterprise_milestone_510_nvlink_infiniband_rocev2_gpudirect_busbw_bible"
    )
    assert data["summary_metrics"]["gpudirect_rdma_busbw_speedup_vs_cpu_bounce"] == 4.23
    assert data["summary_metrics"]["tuned_rocev2_vs_ib_ndr_busbw_parity_pct"] == 96.3
    assert data["summary_metrics"]["ai_ethernet_capex_saving_vs_ib_pct"] == 32.5
    fabrics = data["fabrics"]
    assert len(fabrics) == 5
    assert fabrics[0]["nccl_allreduce_busbw_gbps"] == 11.4
    assert fabrics[2]["nccl_allreduce_busbw_gbps"] == 46.4
    assert fabrics[3]["nccl_allreduce_busbw_gbps"] == 48.2
