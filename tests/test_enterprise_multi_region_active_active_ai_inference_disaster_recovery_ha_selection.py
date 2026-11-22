from pathlib import Path
from experiments.enterprise_multi_region_active_active_ai_inference_disaster_recovery_ha_selection.run import build_results


def test_enterprise_multi_region_active_active_ai_inference_disaster_recovery_ha_selection() -> None:
    res = build_results()
    assert res["experiment_id"] == "enterprise_multi_region_active_active_ai_inference_disaster_recovery_ha_selection"
    tiers = res["dr_tiers"]
    assert len(tiers) == 4
    assert tiers[3]["effective_annual_availability_pct"] >= res["target_sla_nines"]
    assert tiers[3]["failover_rto_seconds"] < 2.0
    assert tiers[3]["standby_capacity_cost_overhead_pct"] < tiers[2]["standby_capacity_cost_overhead_pct"]

    root = Path(__file__).resolve().parents[1]
    assert (root / "docs/components/enterprise-multi-region-active-active-ai-inference-disaster-recovery-ha-selection-faq.md").exists()
    assert (root / "docs/components/enterprise-multi-region-active-active-ai-inference-disaster-recovery-ha-selection-faq.en.md").exists()
    assert (root / "docs/h5/milestone-470-multi-region-active-active-dr-ha-lab.html").exists()
