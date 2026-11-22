from pathlib import Path
from experiments.enterprise_gpu_roofline_dcgm_tensor_dram_active_p99_tail_triage_sizing_selection.run import build_results


def test_enterprise_gpu_roofline_dcgm_tensor_dram_active_p99_tail_triage_sizing_selection() -> None:
    res = build_results()
    assert res["experiment_id"] == "enterprise_gpu_roofline_dcgm_tensor_dram_active_p99_tail_triage_sizing_selection"
    tiers = res["tiers"]
    assert len(tiers) == 4
    assert tiers[3]["mean_time_to_diagnose_min"] < tiers[2]["mean_time_to_diagnose_min"] < tiers[0]["mean_time_to_diagnose_min"]
    assert tiers[3]["root_cause_identification_accuracy_pct"] > tiers[0]["root_cause_identification_accuracy_pct"]

    root = Path(__file__).resolve().parents[1]
    assert (root / "docs/components/enterprise-gpu-roofline-dcgm-tensor-dram-active-p99-tail-triage-sizing-selection-faq.md").exists()
    assert (root / "docs/components/enterprise-gpu-roofline-dcgm-tensor-dram-active-p99-tail-triage-sizing-selection-faq.en.md").exists()
    assert (root / "docs/h5/gpu-roofline-dcgm-tensor-dram-active-tail-triage-lab.html").exists()
