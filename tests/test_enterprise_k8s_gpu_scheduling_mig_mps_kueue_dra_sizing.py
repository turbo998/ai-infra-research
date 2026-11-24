from experiments.enterprise_k8s_gpu_scheduling_mig_mps_kueue_dra_sizing.run import evaluate_k8s_gpu_packing


def test_evaluate_k8s_gpu_packing():
    r = evaluate_k8s_gpu_packing(28, 14, 10)
    assert r["mig_h100_gpus"] == 4
    assert r["timeslice_dev_h100_gpus"] == 2
    assert r["consolidated_total_h100"] == 16
    assert r["whole_gpu_baseline_h100"] == 38
    assert r["gpu_reduction_pct"] == 57.9
    assert r["monthly_savings_usd"] == 50600
