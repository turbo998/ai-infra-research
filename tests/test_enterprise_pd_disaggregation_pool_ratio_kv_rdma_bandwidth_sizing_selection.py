from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_pd_disaggregation_pool_ratio_kv_rdma_bandwidth_sizing() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (
        root
        / "experiments"
        / "enterprise_pd_disaggregation_pool_ratio_kv_rdma_bandwidth_sizing_selection"
        / "run.py"
    )
    subprocess.run([sys.executable, str(script)], check=True)
    results_path = script.parent / "results.json"
    data = json.loads(results_path.read_text(encoding="utf-8"))

    assert (
        data["experiment_id"]
        == "enterprise_pd_disaggregation_pool_ratio_kv_rdma_bandwidth_sizing_selection"
    )
    assert data["summary_metrics"]["rag_32k_gpu_saving_pct_vs_colocated"] == 42.9
    assert data["summary_metrics"]["rag_32k_with_radix_prefix_gpu_saving_pct_vs_colocated"] == 78.6
    assert data["summary_metrics"]["p99_tpot_reduction_factor"] == 4.63
    assert data["kv_architecture_comparison"]["mla_kv_bandwidth_reduction_factor"] == 4.66
    scenarios = data["scenarios"]
    assert len(scenarios) == 5
    assert scenarios[0]["total_h200_gpus"] == 56
    assert scenarios[2]["optimal_pd_ratio"] == "3P : 1D (24P + 8D GPUs)"
    assert scenarios[3]["optimal_pd_ratio"] == "1P : 4D (4P + 16D GPUs)"
    assert scenarios[4]["total_h200_gpus"] == 12
