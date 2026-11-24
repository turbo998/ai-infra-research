from experiments.enterprise_milestone_575_production_ai_platform_patterns_bible_sizing.run import build_summary


def test_enterprise_milestone_575_production_ai_platform_patterns_bible_sizing() -> None:
    s = build_summary()
    opt = s["schemes"]["milestone_575_five_pattern_codesigned_platform_28gpu"]
    assert opt["required_gpus"] == opt["l40s_gpus"] + opt["h100_gpus"] + opt["h200_gpus"] == 28
    assert opt["availability_sla_pct"] == 99.995
    assert opt["sla_goodput_pct"] == 99.4
    assert opt["annual_tco_usd"] == 1520000
    assert s["annual_savings_usd_vs_siloed"] == 5280000
