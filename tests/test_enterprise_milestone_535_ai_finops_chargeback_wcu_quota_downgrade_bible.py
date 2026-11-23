from experiments.enterprise_milestone_535_ai_finops_chargeback_wcu_quota_downgrade_bible.run import build_summary


def test_enterprise_milestone_535_ai_finops_chargeback_wcu_quota_downgrade_bible() -> None:
    s = build_summary()
    assert s["spend_reduction_pct_vs_unmetered"] == 46.2
    assert s["gpu_reduction_pct_vs_unmetered"] == 50.0
    assert s["business_continuity_pct"] == 99.94
    assert s["schemes"]["wcu_accounting_plus_3tier_graceful_downgrade"]["cost_attribution_accuracy_pct"] == 99.6
