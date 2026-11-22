from pathlib import Path
from experiments.enterprise_gpu_power_capping_dvfs_sweetspot_transient_smoothing_tco_selection.run import build_results


def test_enterprise_gpu_power_capping_dvfs_sweetspot_transient_smoothing_tco_selection() -> None:
    res = build_results()
    assert res["experiment_id"] == "enterprise_gpu_power_capping_dvfs_sweetspot_transient_smoothing_tco_selection"
    profs = res["power_profiles"]
    assert len(profs) == 4
    assert profs[3]["gpus_deployable_under_1500kw_it_cap"] > profs[0]["gpus_deployable_under_1500kw_it_cap"]
    assert profs[3]["decode_throughput_retained_pct"] > 98.0

    root = Path(__file__).resolve().parents[1]
    assert (root / "docs/components/enterprise-gpu-power-capping-dvfs-sweetspot-transient-smoothing-tco-selection-faq.md").exists()
    assert (root / "docs/components/enterprise-gpu-power-capping-dvfs-sweetspot-transient-smoothing-tco-selection-faq.en.md").exists()
    assert (root / "docs/h5/gpu-power-capping-dvfs-transient-tco-lab.html").exists()
