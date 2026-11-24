from experiments.enterprise_milestone_580_sa_ten_reference_architectures_bible_sizing.run import build_summary


def test_enterprise_milestone_580_sa_ten_reference_architectures_bible_sizing() -> None:
    s = build_summary()
    opt = s["schemes"]["milestone_580_top10_reference_architectures_28gpu"]
    assert opt["required_gpus"] == opt["l40s_gpus"] + opt["h100_gpus"] + opt["h200_gpus"] == 28
    assert opt["composite_task_accuracy_pct"] == 97.6
    assert opt["cost_per_1m_tokens_usd"] == 0.028
    assert s["cost_reduction_factor_vs_legacy"] == 15.79
    assert s["annual_savings_usd_vs_legacy"] == 8420000
