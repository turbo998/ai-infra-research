from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_llm_parallelism_tp_pp_dp_ep_spec_quant_sizing() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (
        root
        / "experiments"
        / "enterprise_llm_parallelism_tp_pp_dp_ep_spec_quant_sizing_selection"
        / "run.py"
    )
    subprocess.run([sys.executable, str(script)], check=True)
    results_path = script.parent / "results.json"
    data = json.loads(results_path.read_text(encoding="utf-8"))

    assert (
        data["experiment_id"]
        == "enterprise_llm_parallelism_tp_pp_dp_ep_spec_quant_sizing_selection"
    )
    assert data["summary_metrics"]["moe_dp_ep_vs_naive_tp16_throughput_gain"] == 5.68
    assert data["summary_metrics"]["moe_dp_ep_p99_tpot_reduction_pct"] == 70.0
    assert data["summary_metrics"]["mtp_effective_tpot_speedup_factor"] == 2.28
    cfgs = data["configurations"]
    assert len(cfgs) == 4
    assert cfgs[0]["output_tokens_per_sec_per_gpu"] == 680.0
    assert cfgs[3]["output_tokens_per_sec_per_gpu"] == 3860.0
    assert cfgs[3]["effective_tpot_with_mtp_ms"] == 5.8
