from pathlib import Path
from experiments.enterprise_ai_finops_k8s_gpu_opencost_ptu_spot_portfolio_selection.run import build_results


def test_enterprise_ai_finops_k8s_gpu_opencost_ptu_spot_portfolio_selection() -> None:
    res = build_results()
    assert res["experiment_id"] == "enterprise_ai_finops_k8s_gpu_opencost_ptu_spot_portfolio_selection"
    tiers = res["tiers"]
    assert len(tiers) == 4
    assert tiers[3]["monthly_total_bill_usd"] < tiers[0]["monthly_total_bill_usd"]
    assert tiers[3]["chargeback_attribution_error_pct"] < tiers[1]["chargeback_attribution_error_pct"]

    root = Path(__file__).resolve().parents[1]
    assert (root / "docs/components/enterprise-ai-finops-k8s-gpu-opencost-ptu-spot-portfolio-selection-faq.md").exists()
    assert (root / "docs/components/enterprise-ai-finops-k8s-gpu-opencost-ptu-spot-portfolio-selection-faq.en.md").exists()
    assert (root / "docs/h5/ai-finops-k8s-gpu-opencost-ptu-spot-portfolio-lab.html").exists()
