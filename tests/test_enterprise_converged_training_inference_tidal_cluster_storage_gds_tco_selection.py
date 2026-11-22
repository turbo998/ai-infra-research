from pathlib import Path
from experiments.enterprise_converged_training_inference_tidal_cluster_storage_gds_tco_selection.run import build_results


def test_enterprise_converged_training_inference_tidal_cluster_storage_gds_tco_selection() -> None:
    res = build_results()
    assert res["experiment_id"] == "enterprise_converged_training_inference_tidal_cluster_storage_gds_tco_selection"
    tiers = res["tiers"]
    assert len(tiers) == 4
    assert tiers[3]["fleet_24h_effective_utilization_pct"] > tiers[2]["fleet_24h_effective_utilization_pct"] > tiers[0]["fleet_24h_effective_utilization_pct"]
    assert tiers[3]["annual_total_cluster_tco_usd"] < tiers[0]["annual_total_cluster_tco_usd"]

    root = Path(__file__).resolve().parents[1]
    assert (root / "docs/components/enterprise-converged-training-inference-tidal-cluster-storage-gds-tco-selection-faq.md").exists()
    assert (root / "docs/components/enterprise-converged-training-inference-tidal-cluster-storage-gds-tco-selection-faq.en.md").exists()
    assert (root / "docs/h5/milestone-460-converged-tidal-cluster-storage-gds-tco-lab.html").exists()
