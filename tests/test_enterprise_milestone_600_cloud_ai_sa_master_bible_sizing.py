from experiments.enterprise_milestone_600_cloud_ai_sa_master_bible_sizing.run import evaluate_milestone_600


def test_evaluate_milestone_600():
    r = evaluate_milestone_600(1.0)
    assert r["topic_id"] == 600
    assert r["deliverables_synthesized"] == 38
    assert r["unified_total_gpus"] == 52
    assert r["siloed_total_gpus"] == 312
    assert r["gpu_reduction_pct"] == 83.3
    assert r["annual_savings_million_usd"] == 14.84
    assert r["tco_reduction_pct"] == 83.4
