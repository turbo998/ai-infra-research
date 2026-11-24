from experiments.enterprise_ai_observability_otel_langfuse_judge_slo_sizing.run import evaluate_observability_stack


def test_evaluate_observability_stack():
    r = evaluate_observability_stack(100, 42)
    assert r["raw_storage_gb"] == 4200
    assert r["compressed_storage_gb"] == 560
    assert r["langfuse_clickhouse_monthly_usd"] == 1800
    assert r["spot_l40s_judge_gpus"] == 2
    assert r["savings_vs_saas_pct"] == 96.4
