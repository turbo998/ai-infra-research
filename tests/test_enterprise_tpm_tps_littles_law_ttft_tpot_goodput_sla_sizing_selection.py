from pathlib import Path
from experiments.enterprise_tpm_tps_littles_law_ttft_tpot_goodput_sla_sizing_selection.run import build_results


def test_enterprise_tpm_tps_littles_law_ttft_tpot_goodput_sla_sizing_selection() -> None:
    res = build_results()
    assert res["experiment_id"] == "enterprise_tpm_tps_littles_law_ttft_tpot_goodput_sla_sizing_selection"
    tiers = res["tiers"]
    assert len(tiers) == 4
    assert tiers[3]["goodput_sla_attainment_pct"] > tiers[0]["goodput_sla_attainment_pct"]
    assert tiers[3]["monthly_cluster_bill_usd"] < tiers[1]["monthly_cluster_bill_usd"]

    root = Path(__file__).resolve().parents[1]
    assert (root / "docs/components/enterprise-tpm-tps-littles-law-ttft-tpot-goodput-sla-sizing-selection-faq.md").exists()
    assert (root / "docs/components/enterprise-tpm-tps-littles-law-ttft-tpot-goodput-sla-sizing-selection-faq.en.md").exists()
    assert (root / "docs/h5/tpm-tps-littles-law-ttft-tpot-goodput-sla-calculator-lab.html").exists()
