from experiments.enterprise_retail_demand_forecasting_tsfm_gbdt_llm_promo_sizing.run import build_summary


def test_enterprise_retail_demand_forecasting_tsfm_gbdt_llm_promo_sizing() -> None:
    s = build_summary()
    assert s["wape_reduction_pct_vs_arima"] == 45.7
    assert s["wape_reduction_pct_vs_pure_llm"] == 64.1
    assert s["gpu_reduction_pct_vs_pure_llm"] == 93.8
    assert s["schemes"]["llm_promo_features_plus_chronos_plus_gpu_gbdt"]["overall_wape_pct"] == 8.9
