from experiments.enterprise_three_watermark_gpu_sizer_tpm_tps_ptu_littles_law.run import (
    evaluate_three_watermark_gpu_sizer_littles_law,
)


def test_three_watermark_gpu_sizer_littles_law():
    r = evaluate_three_watermark_gpu_sizer_littles_law()
    assert r["deliverable_id"] == "#626"
    assert r["e2el_sec"] == 7.65
    assert r["active_batch_sessions"] == 192
    assert r["final_h200_gpus"] == 8
    assert r["gpu_savings_pct"] == 42.9
    assert r["sized_8gpu_tpot_ms"] == 13.6
    assert r["weighted_ptu_tpm_millions"] == 8.48
    assert r["goodput_pct"] == 99.4
