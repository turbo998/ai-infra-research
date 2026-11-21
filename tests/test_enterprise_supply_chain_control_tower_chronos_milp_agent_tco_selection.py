from pathlib import Path
from experiments.enterprise_supply_chain_control_tower_chronos_milp_agent_tco_selection.run import build_results


def test_enterprise_supply_chain_control_tower_chronos_milp_agent_tco_selection() -> None:
    res = build_results()
    assert res["experiment_id"] == "enterprise_supply_chain_control_tower_chronos_milp_agent_tco_selection"
    tiers = res["tiers"]
    assert len(tiers) == 4
    assert tiers[3]["forecast_wmape_pct"] < tiers[0]["forecast_wmape_pct"]
    assert tiers[3]["hard_constraint_violation_pct"] == 0.0

    root = Path(__file__).resolve().parents[1]
    assert (root / "docs/components/enterprise-supply-chain-control-tower-chronos-milp-agent-tco-selection-faq.md").exists()
    assert (root / "docs/components/enterprise-supply-chain-control-tower-chronos-milp-agent-tco-selection-faq.en.md").exists()
    assert (root / "docs/h5/supply-chain-control-tower-chronos-milp-agent-lab.html").exists()
