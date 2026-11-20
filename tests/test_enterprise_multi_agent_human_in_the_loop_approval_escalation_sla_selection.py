from __future__ import annotations

import unittest
from experiments.enterprise_multi_agent_human_in_the_loop_approval_escalation_sla_selection.run import build_payload


class TestEnterpriseMultiAgentHumanInTheLoopApprovalEscalationSlaSelection(unittest.TestCase):
    def test_tier4_achieves_zero_loss_zero_idle_hold_and_100pct_state_survival(self) -> None:
        payload = build_payload()
        tiers = payload["tiers"]
        self.assertEqual(len(tiers), 4)
        tier1 = tiers[0]
        tier2 = tiers[1]
        tier3 = tiers[2]
        tier4 = tiers[3]
        self.assertEqual(tier4["unauthorized_action_financial_loss_rate_pct"], 0.0)
        self.assertEqual(tier4["idle_worker_or_gpu_memory_hold_pct"], 0.0)
        self.assertEqual(tier4["workflow_state_survival_across_24h_wait_pct"], 100.0)
        self.assertGreater(
            tier4["workflow_state_survival_across_24h_wait_pct"],
            tier2["workflow_state_survival_across_24h_wait_pct"] + 75.0,
        )
        self.assertLess(
            tier4["human_reviewer_fatigue_override_rate_pct"],
            tier2["human_reviewer_fatigue_override_rate_pct"] * 0.05,
        )
        self.assertLess(
            tier4["monthly_500k_workflow_ops_and_risk_tco_usd"],
            tier1["monthly_500k_workflow_ops_and_risk_tco_usd"] * 0.12,
        )

    def test_scenarios_and_perturbations_complete(self) -> None:
        payload = build_payload()
        self.assertEqual(len(payload["scenarios"]), 4)
        self.assertEqual(len(payload["perturbation_checks"]), 2)


if __name__ == "__main__":
    unittest.main()
