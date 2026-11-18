import unittest
from experiments.enterprise_energy_grid_vpp_dispatch_substation_inspection_agent_selection.run import build_payload


class TestEnterpriseEnergyGridVppDispatchSubstationInspectionAgentSelection(unittest.TestCase):
    def test_tier4_achieves_zero_interlock_violations_and_high_forecast_accuracy(self) -> None:
        data = build_payload()
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_manual_trader_bidding_and_manual_substation_inspection"]
        t2 = tiers["tier2_prompt_only_llm_power_bidding_and_unverified_switching_order"]
        t4 = tiers["tier4_weather_llm_stochastic_milp_vpp_plus_radiometric_vlm_plus_five_prevention_agent"]

        self.assertGreater(t4["vpp_renewable_forecast_accuracy_pct"], 93.0)
        self.assertEqual(t4["five_preventions_interlock_violation_rate_pct"], 0.0)
        self.assertGreater(t4["thermal_hotspot_defect_recall_pct"], 96.0)
        self.assertGreater(t2["five_preventions_interlock_violation_rate_pct"], 10.0)
        self.assertLess(t4["monthly_imbalance_penalty_and_ops_cost_usd"], t1["monthly_imbalance_penalty_and_ops_cost_usd"] * 0.2)

    def test_provenance_explicit(self) -> None:
        data = build_payload()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")


if __name__ == "__main__":
    unittest.main()
