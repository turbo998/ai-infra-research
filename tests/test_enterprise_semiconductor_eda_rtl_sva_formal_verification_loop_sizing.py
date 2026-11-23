from experiments.enterprise_semiconductor_eda_rtl_sva_formal_verification_loop_sizing.run import build_summary


def test_enterprise_semiconductor_eda_rtl_sva_formal_verification_loop_sizing() -> None:
    s = build_summary()
    assert s["verification_closure_reduction_pct_vs_manual"] == 72.4
    assert s["functional_pass_gain_pts_vs_zeroshot"] == 49.8
    assert s["schemes"]["fsm_graph_plus_rtl_sva_llm_plus_formal_counterexample_loop"]["synthesizable_rtl_pass_pct"] == 96.8
    assert s["schemes"]["fsm_graph_plus_rtl_sva_llm_plus_formal_counterexample_loop"]["functional_formal_pass_pct"] == 91.4
