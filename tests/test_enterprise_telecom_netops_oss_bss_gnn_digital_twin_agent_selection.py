from pathlib import Path
from experiments.enterprise_telecom_netops_oss_bss_gnn_digital_twin_agent_selection.run import build_results


def test_enterprise_telecom_netops_oss_bss_gnn_digital_twin_agent_selection() -> None:
    res = build_results()
    assert res["experiment_id"] == "enterprise_telecom_netops_oss_bss_gnn_digital_twin_agent_selection"
    tiers = res["tiers"]
    assert len(tiers) == 4
    assert tiers[3]["change_induced_outage_rate_pct"] == 0.0
    assert tiers[3]["mttr_minutes"] < tiers[0]["mttr_minutes"]

    root = Path(__file__).resolve().parents[1]
    assert (root / "docs/components/enterprise-telecom-netops-oss-bss-gnn-digital-twin-agent-selection-faq.md").exists()
    assert (root / "docs/components/enterprise-telecom-netops-oss-bss-gnn-digital-twin-agent-selection-faq.en.md").exists()
    assert (root / "docs/h5/telecom-netops-oss-bss-gnn-digital-twin-agent-lab.html").exists()
