from experiments.enterprise_milestone_545_ai_coding_platform_fim_fastapply_microvm_bible.run import build_summary


def test_enterprise_milestone_545_ai_coding_platform_fim_fastapply_microvm_bible() -> None:
    s = build_summary()
    assert s["fim_latency_speedup_vs_frontier"] == 5.04
    assert s["gpu_reduction_pct_vs_frontier"] == 80.0
    assert s["acceptance_rate_gain_pts_vs_frontier"] == 21.9
    assert s["schemes"]["three_tier_fim_fastapply_microvm_swe"]["required_h200_gpus"] == 32
    assert s["schemes"]["three_tier_fim_fastapply_microvm_swe"]["monthly_cost_per_dev_usd"] == 9.6
