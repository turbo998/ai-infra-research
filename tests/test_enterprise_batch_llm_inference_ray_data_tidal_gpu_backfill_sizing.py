from experiments.enterprise_batch_llm_inference_ray_data_tidal_gpu_backfill_sizing.run import evaluate_batch_tidal_backfill


def test_evaluate_batch_tidal_backfill():
    r = evaluate_batch_tidal_backfill(500, 6)
    assert r["speedup_ratio"] == 4.22
    assert r["completion_hours"] == 4.77
    assert r["max_8h_capacity_million_tokens"] == 838
    assert r["tidal_backfill_monthly_usd"] == 285
    assert r["savings_vs_cloud_batch_pct"] == 97.5
