from __future__ import annotations

import unittest
from experiments.enterprise_agentic_orchestration_langgraph_temporal_autogen_foundry_selection.run import build_payload


class TestEnterpriseAgenticOrchestrationLanggraphTemporalAutogenFoundrySelection(unittest.TestCase):
    def test_tier4_zero_restart_loss_zero_duplicate_side_effects_and_low_tco(self) -> None:
        payload = build_payload()
        tiers = payload["tiers"]
        self.assertEqual(len(tiers), 4)
        tier1 = tiers[0]
        tier4 = tiers[3]
        self.assertEqual(tier4["pod_restart_mid_workflow_loss_pct"], 0.0)
        self.assertEqual(tier4["duplicate_side_effect_tool_execution_pct"], 0.0)
        self.assertEqual(tier4["multi_day_hitl_approval_Idle_compute_waste_pct"], 0.0)
        self.assertEqual(tier4["workflow_observability_and_replay_recovery_pct"], 100.0)
        self.assertLess(
            tier4["monthly_platform_ops_and_failure_rework_tco_usd"],
            tier1["monthly_platform_ops_and_failure_rework_tco_usd"] * 0.30,
        )

    def test_scenarios_and_perturbations_complete(self) -> None:
        payload = build_payload()
        self.assertEqual(len(payload["scenarios"]), 4)
        self.assertEqual(len(payload["perturbation_checks"]), 2)


if __name__ == "__main__":
    unittest.main()
