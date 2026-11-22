from pathlib import Path
from experiments.enterprise_gpu_partitioning_mig_mps_timeslicing_small_model_packing_selection.run import build_results


def test_enterprise_gpu_partitioning_mig_mps_timeslicing_small_model_packing_selection() -> None:
    res = build_results()
    assert res["experiment_id"] == "enterprise_gpu_partitioning_mig_mps_timeslicing_small_model_packing_selection"
    modes = res["partition_modes"]
    assert len(modes) == 4
    assert modes[3]["gpus_required_for_14_microservices"] == 2
    assert modes[3]["noisy_neighbor_p99_jitter_pct"] < 2.0
    assert modes[3]["hardware_fault_and_oom_isolation"] is True

    root = Path(__file__).resolve().parents[1]
    assert (root / "docs/components/enterprise-gpu-partitioning-mig-mps-timeslicing-small-model-packing-selection-faq.md").exists()
    assert (root / "docs/components/enterprise-gpu-partitioning-mig-mps-timeslicing-small-model-packing-selection-faq.en.md").exists()
    assert (root / "docs/h5/gpu-partitioning-mig-mps-timeslicing-packing-lab.html").exists()
