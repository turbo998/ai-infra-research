from pathlib import Path
from experiments.enterprise_day0_day2_ai_cluster_acceptance_burnin_chaos_golive_selection.run import build_results


def test_enterprise_day0_day2_ai_cluster_acceptance_burnin_chaos_golive_selection() -> None:
    res = build_results()
    assert res["experiment_id"] == "enterprise_day0_day2_ai_cluster_acceptance_burnin_chaos_golive_selection"
    tiers = res["acceptance_tiers"]
    assert len(tiers) == 4
    assert tiers[3]["doa_and_lemon_nodes_caught_before_prod_pct"] > 99.0
    assert tiers[3]["first_month_production_job_abort_rate_pct"] < 1.0

    root = Path(__file__).resolve().parents[1]
    assert (root / "docs/components/enterprise-day0-day2-ai-cluster-acceptance-burnin-chaos-golive-selection-faq.md").exists()
    assert (root / "docs/components/enterprise-day0-day2-ai-cluster-acceptance-burnin-chaos-golive-selection-faq.en.md").exists()
    assert (root / "docs/h5/milestone-485-day0-day2-cluster-acceptance-golive-lab.html").exists()
