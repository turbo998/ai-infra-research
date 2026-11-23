from experiments.enterprise_milestone_550_fullstack_ai_infra_master_compendium_sizing.run import build_summary


def test_enterprise_milestone_550_fullstack_ai_infra_master_compendium_sizing() -> None:
    s = build_summary()
    assert s["token_cost_speedup_vs_siloed"] == 14.48
    assert s["gpu_reduction_pct_vs_siloed"] == 87.5
    assert s["annual_savings_usd_vs_siloed"] == 7820000
    assert s["schemes"]["fullstack_codesigned_l0_to_l4_architecture"]["cost_per_1m_tokens_usd"] == 0.029
