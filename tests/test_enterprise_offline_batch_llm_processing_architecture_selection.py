from __future__ import annotations

import unittest
from experiments.enterprise_offline_batch_llm_processing_architecture_selection.run import build_payload


class TestEnterpriseOfflineBatchLlmProcessingArchitectureSelection(unittest.TestCase):
    def test_tier4_lowest_unit_cost_fastest_completion_and_zero_online_degradation(self) -> None:
        payload = build_payload()
        tiers = payload["tiers"]
        self.assertEqual(len(tiers), 4)
        tier1 = tiers[0]
        tier2 = tiers[1]
        tier4 = tiers[3]
        self.assertLess(tier4["effective_cost_per_million_tokens_usd"], tier2["effective_cost_per_million_tokens_usd"] * 0.35)
        self.assertEqual(tier4["online_copilot_sla_degradation_rate_pct"], 0.0)
        self.assertLess(tier4["nightly_10m_record_batch_completion_hours"], 4.0)
        self.assertGreater(tier4["gpu_and_ptu_24x7_utilization_efficiency_pct"], 94.0)
        self.assertLess(
            tier4["monthly_batch_processing_tco_usd"],
            tier1["monthly_batch_processing_tco_usd"] * 0.16,
        )

    def test_scenarios_and_perturbations_complete(self) -> None:
        payload = build_payload()
        self.assertEqual(len(payload["scenarios"]), 4)
        self.assertEqual(len(payload["perturbation_checks"]), 2)


if __name__ == "__main__":
    unittest.main()
