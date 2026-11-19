from __future__ import annotations

import unittest
from experiments.enterprise_coding_assistant_swe_agent_platform_selection.run import build_payload


class TestEnterpriseCodingAssistantSweAgentPlatformSelection(unittest.TestCase):
    def test_tier4_highest_pr_resolution_fast_fim_zero_vuln_escape_and_lowest_tco(self) -> None:
        payload = build_payload()
        tiers = payload["tiers"]
        self.assertEqual(len(tiers), 4)
        tier1 = tiers[0]
        tier3 = tiers[2]
        tier4 = tiers[3]
        self.assertGreater(tier4["multi_file_pr_task_resolution_rate_pct"], 84.0)
        self.assertLessEqual(tier4["ide_inline_completion_p95_latency_ms"], 130.0)
        self.assertLess(
            tier4["ide_inline_completion_p95_latency_ms"],
            tier3["ide_inline_completion_p95_latency_ms"] * 0.20,
        )
        self.assertEqual(tier4["ai_generated_vulnerability_and_secret_escape_pct"], 0.0)
        self.assertLess(tier4["hallucinated_internal_api_or_package_rate_pct"], 1.0)
        self.assertLess(
            tier4["monthly_net_engineering_productivity_tco_usd"],
            tier1["monthly_net_engineering_productivity_tco_usd"] * 0.30,
        )

    def test_scenarios_and_perturbations_complete(self) -> None:
        payload = build_payload()
        self.assertEqual(len(payload["scenarios"]), 4)
        self.assertEqual(len(payload["perturbation_checks"]), 2)


if __name__ == "__main__":
    unittest.main()
