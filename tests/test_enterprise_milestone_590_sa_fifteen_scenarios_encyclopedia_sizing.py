from experiments.enterprise_milestone_590_sa_fifteen_scenarios_encyclopedia_sizing.run import evaluate_milestone_590


def test_evaluate_milestone_590():
    r = evaluate_milestone_590(1.0)
    assert r["deliverables_synthesized"] == 28
    assert r["unified_total_gpus"] == 46
    assert r["siloed_total_gpus"] == 248
    assert r["gpu_reduction_pct"] == 81.5
    assert r["annual_savings_million_usd"] == 11.56
    assert r["tco_reduction_pct"] == 81.4
