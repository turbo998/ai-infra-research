from experiments.enterprise_edtech_stem_latex_sympy_socratic_tutor_sizing.run import build_summary


def test_enterprise_edtech_stem_latex_sympy_socratic_tutor_sizing() -> None:
    s = build_summary()
    assert s["latency_speedup_vs_70b_vlm"] == 10.0
    assert s["gpu_reduction_pct_vs_70b_vlm"] == 87.5
    assert s["step_accuracy_gain_pts_vs_70b_vlm"] == 19.0
    assert s["schemes"]["vision_latex_plus_sympy_verifier_plus_socratic_8b"]["step_grading_accuracy_pct"] == 99.6
