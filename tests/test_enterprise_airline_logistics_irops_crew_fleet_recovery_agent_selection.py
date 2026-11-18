import unittest
from experiments.enterprise_airline_logistics_irops_crew_fleet_recovery_agent_selection.run import build_payload


class TestEnterpriseAirlineLogisticsIropsCrewFleetRecoveryAgentSelection(unittest.TestCase):
    def test_tier4_maximizes_recovery_with_zero_far117_crew_violations(self) -> None:
        data = build_payload()
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_manual_occ_dispatcher_and_sequential_siloed_recovery"]
        t2 = tiers["tier2_prompt_only_llm_direct_tail_and_crew_rescheduling"]
        t4 = tiers["tier4_notam_llm_parser_plus_integrated_column_generation_solver_plus_far117_gate"]

        self.assertGreater(t4["passenger_and_cargo_on_time_recovery_pct"], 92.0)
        self.assertEqual(t4["crew_far117_duty_legality_violation_rate_pct"], 0.0)
        self.assertGreater(t2["crew_far117_duty_legality_violation_rate_pct"], 20.0)
        self.assertLessEqual(t4["schedule_recovery_computation_minutes"], 5.0)
        self.assertLess(t4["monthly_irops_cancellation_and_delay_cost_usd"], t1["monthly_irops_cancellation_and_delay_cost_usd"] * 0.2)

    def test_provenance_explicit(self) -> None:
        data = build_payload()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")


if __name__ == "__main__":
    unittest.main()
