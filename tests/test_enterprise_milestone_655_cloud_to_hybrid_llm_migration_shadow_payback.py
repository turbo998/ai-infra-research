from experiments.enterprise_milestone_655_cloud_to_hybrid_llm_migration_shadow_payback.run import (
    evaluate_cloud_to_hybrid_migration,
)


def test_enterprise_milestone_655_cloud_to_hybrid_llm_migration_shadow_payback() -> None:
    res = evaluate_cloud_to_hybrid_migration()
    assert res["Deliverable_ID"] == "#655"
    assert res["Parity_Alignment"]["lora_plus_xgrammar_parity_pct"] == 99.3
    fin = res["FinOps_Payback"]
    assert fin["monthly_saving_pct"] == 69.7
    assert fin["payback_period_months"] == 5.4
    assert fin["three_year_net_saving_usd"] == 3840000
