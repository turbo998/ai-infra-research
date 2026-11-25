from experiments.enterprise_k8s_mig_mps_kueue_coldstart_elastic_sizing.run import (
    evaluate_k8s_mig_mps_coldstart_sizing,
)


def test_k8s_mig_mps_coldstart_sizing():
    r = evaluate_k8s_mig_mps_coldstart_sizing()
    assert r["deliverable_id"] == "#634"
    assert r["mig_1g18gb_gpu_count"] == 4
    assert r["fast_streamer_coldstart_sec"] == 14.2
    assert r["coldstart_speedup_x"] == 35.9
    assert r["optimized_total_h200"] == 24
    assert r["gpu_savings_pct"] == 62.5
    assert r["annual_savings_usd"] == 1536000
