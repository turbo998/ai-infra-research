from experiments.enterprise_milestone_560_18_industry_vertical_sa_architecture_bible.run import build_summary


def test_enterprise_milestone_560_18_industry_vertical_sa_architecture_bible() -> None:
    s = build_summary()
    assert s["covered_verticals"] == 18
    assert s["average_gpu_reduction_pct_vs_monolithic"] == 86.4
    assert s["domain_compliance_gain_pts_vs_monolithic"] == 31.4
    assert s["schemes"]["neuro_symbolic_cascade_plus_deterministic_solver"]["domain_invariant_compliance_pct"] == 99.8
