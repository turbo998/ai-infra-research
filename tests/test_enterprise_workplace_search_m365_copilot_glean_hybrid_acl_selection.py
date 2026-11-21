from pathlib import Path
from experiments.enterprise_workplace_search_m365_copilot_glean_hybrid_acl_selection.run import build_results


def test_enterprise_workplace_search_m365_copilot_glean_hybrid_acl_selection() -> None:
    res = build_results()
    assert res["experiment_id"] == "enterprise_workplace_search_m365_copilot_glean_hybrid_acl_selection"
    tiers = res["tiers"]
    assert len(tiers) == 4
    assert tiers[3]["permission_aware_recall_at_5_pct"] > tiers[0]["permission_aware_recall_at_5_pct"]
    assert tiers[3]["monthly_tco_usd"] < tiers[1]["monthly_tco_usd"]

    root = Path(__file__).resolve().parents[1]
    assert (root / "docs/components/enterprise-workplace-search-m365-copilot-glean-hybrid-acl-selection-faq.md").exists()
    assert (root / "docs/components/enterprise-workplace-search-m365-copilot-glean-hybrid-acl-selection-faq.en.md").exists()
    assert (root / "docs/h5/workplace-search-m365-copilot-glean-hybrid-acl-lab.html").exists()
