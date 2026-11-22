from pathlib import Path
from experiments.enterprise_end_to_end_ai_infra_sizing_pd_parallelism_network_tco_blueprint_selection.run import build_results


def test_enterprise_end_to_end_ai_infra_sizing_pd_parallelism_network_tco_blueprint_selection() -> None:
    res = build_results()
    assert res["experiment_id"] == "enterprise_end_to_end_ai_infra_sizing_pd_parallelism_network_tco_blueprint_selection"
    tiers = res["tiers"]
    assert len(tiers) == 4
    assert tiers[3]["cluster_goodput_tok_s"] > tiers[2]["cluster_goodput_tok_s"] > tiers[0]["cluster_goodput_tok_s"]
    assert tiers[3]["three_year_total_tco_usd"] < tiers[0]["three_year_total_tco_usd"]

    root = Path(__file__).resolve().parents[1]
    assert (root / "docs/components/enterprise-end-to-end-ai-infra-sizing-pd-parallelism-network-tco-blueprint-selection-faq.md").exists()
    assert (root / "docs/components/enterprise-end-to-end-ai-infra-sizing-pd-parallelism-network-tco-blueprint-selection-faq.en.md").exists()
    assert (root / "docs/h5/milestone-450-end-to-end-ai-infra-sizing-blueprint-lab.html").exists()
