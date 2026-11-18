import unittest
from experiments.enterprise_cold_chain_logistics_vrptw_fleet_dispatch_agent_selection.run import build_payload


class TestEnterpriseColdChainLogisticsVrptwFleetDispatchAgentSelection(unittest.TestCase):
    def test_tier4_maximizes_ontime_delivery_with_zero_coldchain_spoilage(self) -> None:
        data = build_payload()
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_manual_zone_dispatch_and_post_trip_usb_logger"]
        t2 = tiers["tier2_prompt_only_llm_direct_route_sequencing"]
        t4 = tiers["tier4_llm_dock_rule_compiler_plus_dynamic_vrptw_solver_plus_thermal_inertia_rescue_agent"]

        self.assertGreater(t4["on_time_in_window_delivery_rate_pct"], 97.0)
        self.assertEqual(t4["cold_chain_thermal_spoilage_rate_pct"], 0.0)
        self.assertLess(t2["on_time_in_window_delivery_rate_pct"], t1["on_time_in_window_delivery_rate_pct"])
        self.assertLessEqual(t4["empty_deadhead_mileage_ratio_pct"], 13.0)
        self.assertLess(t4["monthly_fuel_spoilage_and_penalty_cost_usd"], t1["monthly_fuel_spoilage_and_penalty_cost_usd"] * 0.2)

    def test_provenance_explicit(self) -> None:
        data = build_payload()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")


if __name__ == "__main__":
    unittest.main()
