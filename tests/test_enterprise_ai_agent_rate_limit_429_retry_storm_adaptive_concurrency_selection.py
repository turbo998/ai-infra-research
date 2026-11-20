from __future__ import annotations

import unittest
from experiments.enterprise_ai_agent_rate_limit_429_retry_storm_adaptive_concurrency_selection.run import build_payload


class TestEnterpriseAiAgentRateLimit429RetryStormAdaptiveConcurrencySelection(unittest.TestCase):
    def test_tier4_eliminates_retry_amplification_and_maximizes_surge_completion(self) -> None:
        payload = build_payload()
        tiers = payload["tiers"]
        self.assertEqual(len(tiers), 4)
        tier1 = tiers[0]
        tier2 = tiers[1]
        tier3 = tiers[2]
        tier4 = tiers[3]
        self.assertLessEqual(tier4["traffic_amplification_factor_under_overload"], 1.10)
        self.assertLess(
            tier4["brownout_recovery_time_after_spike_seconds"],
            tier2["brownout_recovery_time_after_spike_seconds"] * 0.05,
        )
        self.assertGreater(tier4["successful_workflow_completion_under_2x_surge_pct"], 99.0)
        self.assertGreater(
            tier4["successful_workflow_completion_under_2x_surge_pct"],
            tier1["successful_workflow_completion_under_2x_surge_pct"] + 75.0,
        )
        self.assertLess(
            tier4["monthly_incident_and_burst_overage_tco_usd"],
            tier3["monthly_incident_and_burst_overage_tco_usd"] * 0.30,
        )

    def test_scenarios_and_perturbations_complete(self) -> None:
        payload = build_payload()
        self.assertEqual(len(payload["scenarios"]), 4)
        self.assertEqual(len(payload["perturbation_checks"]), 2)


if __name__ == "__main__":
    unittest.main()
