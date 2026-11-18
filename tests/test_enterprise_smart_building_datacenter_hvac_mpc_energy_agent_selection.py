import unittest
from experiments.enterprise_smart_building_datacenter_hvac_mpc_energy_agent_selection.run import build_payload


class TestEnterpriseSmartBuildingDatacenterHvacMpcEnergyAgentSelection(unittest.TestCase):
    def test_tier4_maximizes_hvac_savings_with_zero_thermal_or_surge_violations(self) -> None:
        data = build_payload()
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_legacy_bms_fixed_schedules_and_reactive_pid"]
        t2 = tiers["tier2_prompt_only_llm_direct_bacnet_setpoint_control"]
        t4 = tiers["tier4_llm_brick_ontology_mapper_plus_physics_mpc_solver_plus_bacnet_safety_clamp_agent"]

        self.assertGreater(t4["hvac_energy_and_peak_demand_savings_pct"], 24.0)
        self.assertEqual(t4["ashrae_thermal_or_chiller_surge_violation_rate_pct"], 0.0)
        self.assertGreater(t2["ashrae_thermal_or_chiller_surge_violation_rate_pct"], 10.0)
        self.assertLessEqual(t4["bms_point_onboarding_hours_per_building"], 6.0)
        self.assertLess(t4["monthly_hvac_electricity_and_ops_cost_usd"], t1["monthly_hvac_electricity_and_ops_cost_usd"] * 0.2)

    def test_provenance_explicit(self) -> None:
        data = build_payload()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")


if __name__ == "__main__":
    unittest.main()
