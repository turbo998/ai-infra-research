from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_gpu_three_watermark_sizer_tpm_tps_ptu_sla() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (
        root
        / "experiments"
        / "enterprise_gpu_three_watermark_sizer_tpm_tps_ptu_sla_selection"
        / "run.py"
    )
    subprocess.run([sys.executable, str(script)], check=True)
    results_path = script.parent / "results.json"
    data = json.loads(results_path.read_text(encoding="utf-8"))

    assert (
        data["experiment_id"]
        == "enterprise_gpu_three_watermark_sizer_tpm_tps_ptu_sla_selection"
    )
    assert data["summary_metrics"]["goodput_improvement_pts_vs_weights_only"] == 80.7
    assert data["summary_metrics"]["gpu_saving_pct_vs_colocated_three_watermark"] == 40.0
    assert data["tpm_tps_conversion_example"]["input_tpm"] == 14745600.0
    assert data["tpm_tps_conversion_example"]["output_tpm"] == 921600.0
    methods = data["sizing_methods"]
    assert len(methods) == 4
    assert methods[0]["sla_goodput_pct"] == 18.5
    assert methods[3]["provisioned_h200_gpus"] == 12
    assert methods[3]["sla_goodput_pct"] == 99.2
