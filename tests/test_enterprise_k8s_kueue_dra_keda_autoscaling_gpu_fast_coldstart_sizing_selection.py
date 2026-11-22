from pathlib import Path
from experiments.enterprise_k8s_kueue_dra_keda_autoscaling_gpu_fast_coldstart_sizing_selection.run import build_results


def test_enterprise_k8s_kueue_dra_keda_autoscaling_gpu_fast_coldstart_sizing_selection() -> None:
    res = build_results()
    assert res["experiment_id"] == "enterprise_k8s_kueue_dra_keda_autoscaling_gpu_fast_coldstart_sizing_selection"
    tiers = res["tiers"]
    assert len(tiers) == 4
    assert tiers[3]["replica_cold_start_sec"] < tiers[2]["replica_cold_start_sec"] < tiers[0]["replica_cold_start_sec"]
    assert tiers[3]["cluster_24h_utilization_pct"] > tiers[0]["cluster_24h_utilization_pct"]

    root = Path(__file__).resolve().parents[1]
    assert (root / "docs/components/enterprise-k8s-kueue-dra-keda-autoscaling-gpu-fast-coldstart-sizing-selection-faq.md").exists()
    assert (root / "docs/components/enterprise-k8s-kueue-dra-keda-autoscaling-gpu-fast-coldstart-sizing-selection-faq.en.md").exists()
    assert (root / "docs/h5/k8s-kueue-dra-keda-gpu-fast-coldstart-sizer-lab.html").exists()
