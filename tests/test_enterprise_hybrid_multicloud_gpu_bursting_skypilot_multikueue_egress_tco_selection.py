from pathlib import Path
from experiments.enterprise_hybrid_multicloud_gpu_bursting_skypilot_multikueue_egress_tco_selection.run import build_results


def test_enterprise_hybrid_multicloud_gpu_bursting_skypilot_multikueue_egress_tco_selection() -> None:
    res = build_results()
    assert res["experiment_id"] == "enterprise_hybrid_multicloud_gpu_bursting_skypilot_multikueue_egress_tco_selection"
    models = res["bursting_models"]
    assert len(models) == 4
    assert models[3]["annual_total_tco_usd"] < models[0]["annual_total_tco_usd"] * 0.45
    assert models[3]["cloud_burst_egress_bill_monthly_usd"] < 1000.0

    root = Path(__file__).resolve().parents[1]
    assert (root / "docs/components/enterprise-hybrid-multicloud-gpu-bursting-skypilot-multikueue-egress-tco-selection-faq.md").exists()
    assert (root / "docs/components/enterprise-hybrid-multicloud-gpu-bursting-skypilot-multikueue-egress-tco-selection-faq.en.md").exists()
    assert (root / "docs/h5/hybrid-multicloud-gpu-bursting-egress-tco-lab.html").exists()
