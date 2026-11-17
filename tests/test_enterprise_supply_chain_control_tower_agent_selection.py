import unittest
from experiments.enterprise_supply_chain_control_tower_agent_selection.run import build_Payload


class TestEnterpriseSupplyChainControlTowerAgentSelection(unittest.TestCase):
    def test_tier4_maximizes_forecast_accuracy_and_zero_constraint_violations(self) -> None:
        data = build_Payload()
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_legacy_erp_moving_average_manual_excel"]
        t2 = tiers["tier2_pure_llm_text_time_series_forecaster"]
        t4 = tiers["tier4_llm_shock_covariates_chronos_ortools_milp_control_tower"]

        self.assertGreater(t4["forecast_accuracy_1_minus_wape_pct"], 90.0)
        self.assertLess(t2["forecast_accuracy_1_minus_wape_pct"], t1["forecast_accuracy_1_minus_wape_pct"])
        self.assertEqual(t4["moq_capacity_violation_rate_pct"], 0.0)
        self.assertLess(t4["monthly_total_cost_usd"], t1["monthly_total_cost_usd"] * 0.2)

    def test_provenance_Explicit(self) -> None:
        data = build_Payload()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")


if __name__ == "__main__":
    unittest.main()
