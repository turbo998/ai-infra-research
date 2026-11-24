from experiments.enterprise_milestone_570_sa_four_pillars_master_bible_sizing.run import build_summary


def test_enterprise_milestone_570_sa_four_pillars_master_bible_sizing() -> None:
    s = build_summary()
    opt = s["schemes"]["milestone_570_four_pillar_codesigned_architecture"]
    assert opt["required_gpus"] == opt["prefill_h100_gpus"] + opt["decode_h200_gpus"] == 20
    assert opt["sla_goodput_pct"] == 99.4
    assert opt["cost_per_1m_goodput_tokens_usd"] == 0.038
    assert s["cost_reduction_factor_vs_legacy"] == 11.79
    assert s["annual_savings_usd_vs_legacy"] == 3840000
