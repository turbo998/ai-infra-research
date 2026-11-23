from experiments.enterprise_continuous_llm_eval_cicd_async_shadow_judge_sizing.run import build_summary


def test_enterprise_continuous_llm_eval_cicd_async_shadow_judge_sizing() -> None:
    s = build_summary()
    assert s["eval_gpu_reduction_pct_vs_inline"] == 100.0
    assert s["monthly_eval_cost_reduction_pct_vs_inline"] == 98.3
    assert s["schemes"]["cicd_500case_plus_2ms_drift_plus_2pct_async_shadow"]["regression_detection_recall_pct"] == 97.8
    assert s["schemes"]["cicd_500case_plus_2ms_drift_plus_2pct_async_shadow"]["added_online_p99_latency_ms"] == 0.0
