from experiments.enterprise_ai_finops_wcu_reserved_spot_paygo_portfolio_sizing.run import build_summary


def test_enterprise_ai_finops_wcu_reserved_spot_paygo_portfolio_sizing() -> None:
    s = build_summary()
    opt = s["schemes"]["wcu_3tier_portfolio_graceful_downgrade"]
    assert s["wcu_weights"]["output_decode_token"] == 4.5
    assert opt["attribution_accuracy_pct"] == 99.6
    assert opt["service_continuity_pct"] == 99.96
    assert opt["monthly_spend_usd"] == 95000
    assert s["annual_savings_usd_vs_flat"] == 1380000
