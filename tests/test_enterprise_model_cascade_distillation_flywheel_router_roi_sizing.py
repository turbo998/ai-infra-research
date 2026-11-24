from experiments.enterprise_model_cascade_distillation_flywheel_router_roi_sizing.run import build_summary


def test_enterprise_model_cascade_distillation_flywheel_router_roi_sizing() -> None:
    s = build_summary()
    opt = s["schemes"]["pre_routed_5ms_82_18_distilled_8b_70b_cascade"]
    assert s["distillation_economics"]["breakeven_hours"] == 4.2
    assert opt["task_accuracy_pct"] == 98.9
    assert opt["p95_ttft_ms"] == 145.0
    assert opt["required_h200_gpus"] == 12
    assert s["annual_savings_usd_vs_all_70b"] == 1296000
