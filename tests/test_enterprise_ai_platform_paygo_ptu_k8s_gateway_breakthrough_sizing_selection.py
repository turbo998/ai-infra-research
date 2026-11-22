from pathlib import Path
from experiments.enterprise_ai_platform_paygo_ptu_k8s_gateway_breakthrough_sizing_selection.run import build_results


def test_enterprise_ai_platform_paygo_ptu_k8s_gateway_breakthrough_sizing_selection() -> None:
    res = build_results()
    assert res["experiment_id"] == "enterprise_ai_platform_paygo_ptu_k8s_gateway_breakthrough_sizing_selection"
    tiers = res["tiers"]
    assert len(tiers) == 4
    assert tiers[3]["annual_total_tco_usd"] < tiers[2]["annual_total_tco_usd"] < tiers[0]["annual_total_tco_usd"]
    assert tiers[3]["prefix_kv_cache_hit_pct"] > tiers[0]["prefix_kv_cache_hit_pct"]

    root = Path(__file__).resolve().parents[1]
    assert (root / "docs/components/enterprise-ai-platform-paygo-ptu-k8s-gateway-breakthrough-sizing-selection-faq.md").exists()
    assert (root / "docs/components/enterprise-ai-platform-paygo-ptu-k8s-gateway-breakthrough-sizing-selection-faq.en.md").exists()
    assert (root / "docs/h5/enterprise-ai-platform-paygo-ptu-k8s-gateway-breakthrough-lab.html").exists()
